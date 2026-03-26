import os
import httpx
import edge_tts
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_AI = "llama-3.1-8b-instant"

bisedat: Dict[str, List[Dict]] = {}

# --- FUNKSIONI PËR STT (ZËRI NË TEKST) ---
async def speech_to_text(audio_data):
    async with httpx.AsyncClient() as client:
        files = {'file': ('audio.wav', audio_data, 'audio/wav')}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        # Përdorim modelin më të shpejtë Whisper
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data={"model": "whisper-large-v3-turbo", "language": "sq"}
        )
        return response.json().get("text", "")

@app.post("/listen")
async def listen(file: UploadFile = File(...), device_id: str = Form(...)):
    audio_bytes = await file.read()
    user_text = await speech_to_text(audio_bytes)
    
    if not user_text:
        return {"answer": "Më fal, nuk të dëgjova mirë.", "following_mode": False}

    # Këtu thërrasim logjikën tënde të bisedës (ask)
    # ... (kodi yt ekzistues që gjeneron përgjigjen me Llama 3.1)
    
    # Supozojmë se përgjigja është 'pergjigja_ai'
    pergjigja_ai = "Përshëndetje Noel! Jam gati." # Kjo vjen nga Groq
    
    # Gjenerojmë audion për boksin
    communicate = edge_tts.Communicate(pergjigja_ai, "sq-AL-AlbaNeural")
    audio_output = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_output += chunk["data"]

    # Dërgojmë audion mbrapsht te Luna
    return {
        "text": pergjigja_ai,
        "audio": audio_output.hex(), # E dërgojmë si hex që ESP ta kuptojë
        "following_mode": True # I thotë Lunës rri hapur për bisedë
    }
