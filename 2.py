import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# dotenv load करें
load_dotenv()

app = FastAPI(title="Gemini AI Assistant Workspace Backend")

# React frontend से connect करने के लिए CORS allow करें
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # local testing के लिए '*' रख सकते हैं
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema definitions
class Part(BaseModel):
    text: str

class Content(BaseModel):
    role: str
    parts: List[Part]

class ChatRequest(BaseModel):
    contents: List[Content]
    systemInstruction: Optional[str] = "You are a professional AI Assistant."
    temperature: Optional[float] = 0.7
    model: Optional[str] = "gemini-2.5-flash"

# Global Client Initialization (ताकि हर रिक्वेस्ट पर बार-बार क्लाइंट न बने)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable is missing. Please configure your .env file.")
    ai_client = None
else:
    ai_client = genai.Client(api_key=api_key)


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    if not ai_client:
         raise HTTPException(
            status_code=500, 
            detail="Server is misconfigured: GEMINI_API_KEY is missing."
        )

    try:
        # User payload data structure parsing for genai SDK format
        formatted_contents = []
        for c in payload.contents:
            parts_list = [types.Part.from_text(text=p.text) for p in c.parts]
            formatted_contents.append(
                types.Content(role=c.role, parts=parts_list)
            )

        # Fallback list in case model is overloaded (503/429 errors)
        primary_model = payload.model
        fallbacks = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"]
        models_to_try = [primary_model] + [m for m in fallbacks if m != primary_model]

        response = None
        used_model = primary_model
        fallback_triggered = False

        for current_model in models_to_try:
            try:
                # API Call using modern SDK
                response = ai_client.models.generate_content(
                    model=current_model,
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=payload.systemInstruction,
                        temperature=payload.temperature,
                    )
                )
                used_model = current_model
                if current_model != primary_model:
                    fallback_triggered = True
                break # Success! Break out of the fallback retry loop
            except Exception as e:
                err_msg = str(e)
                print(f"[Gemini Try] Model {current_model} failed. Error: {err_msg}")
                # Transients checking (503/429 status or rate limit indicators)
                if "503" not in err_msg and "429" not in err_msg and "UNAVAILABLE" not in err_msg and "overload" not in err_msg:
                    raise HTTPException(status_code=400, detail=f"Bad request: {err_msg}")
                # Loop will automatically try next fallback model
        
        if response:
            text_result = response.text or "I was unable to formulate a response."
            if fallback_triggered:
                text_result = f"*(Note: The primary model was busy, request routed to fallback: **{used_model}**)*\n\n" + text_result
            
            return {
                "text": text_result,
                "model": used_model
            }
        else:
            raise HTTPException(status_code=503, detail="All model retries were exhausted.")

    except HTTPException:
        # HTTPException को सीधा raise करें ताकि 400 या 503 का स्टेटस कोड 500 में न बदले
        raise 
    except Exception as e:
        print(f"Server error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 to listen on all interfaces or 127.0.0.1 for local
    uvicorn.run(app, host="127.0.0.1", port=3000)