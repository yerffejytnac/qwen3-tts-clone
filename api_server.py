"""
FastAPI server for TTS with streaming audio response
Supports both streaming and non-streaming modes for JavaScript clients
"""

import io
import pickle
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    x_vector_only: bool = False
    max_new_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 50


class TTSServer:
    def __init__(self, model_path: str, voice_model_path: str):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing TTS server on {self.device}...")

        print(f"Loading TTS model: {model_path}")
        self.tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.float32,
        )
        print("✓ TTS model loaded")

        print(f"Loading voice model: {voice_model_path}")
        with open(voice_model_path, "rb") as f:
            self.voice_model = pickle.load(f)
        print("✓ Voice model loaded\n")

    def generate_audio(self, request: TTSRequest) -> tuple[np.ndarray, int]:
        """Generate audio from text"""
        gen_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": True,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
            "subtalker_dosample": True,
            "subtalker_top_k": request.subtalker_top_k,
            "subtalker_top_p": request.subtalker_top_p,
            "subtalker_temperature": request.subtalker_temperature,
            "non_streaming_mode": False,
        }

        wavs, sr = self.tts.generate_voice_clone(
            text=request.text,
            language=request.language,
            voice_clone_prompt=self.voice_model,
            x_vector_only_mode=request.x_vector_only,
            **gen_kwargs,
        )

        return wavs[0], sr


# Initialize server
app = FastAPI(title="Qwen3-TTS API", version="1.0.0")

# Enable CORS for JavaScript clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS instance
tts_server: Optional[TTSServer] = None


@app.on_event("startup")
async def startup_event():
    global tts_server
    import os

    model_path = os.path.expanduser("~/LLMs/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base")
    voice_model_path = "models/jeffrey_voice.pkl"
    tts_server = TTSServer(model_path, voice_model_path)


@app.get("/")
async def root():
    return {
        "name": "Qwen3-TTS API",
        "version": "1.0.0",
        "endpoints": {
            "POST /tts": "Generate speech from text (returns WAV file)",
            "POST /tts/stream": "Generate speech with streaming (returns WAV file)",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "device": tts_server.device if tts_server else "not loaded"}


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """
    Generate speech and return complete WAV file
    JavaScript usage:
        const response = await fetch('http://localhost:8000/tts', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: 'Hello world', language: 'English'})
        });
        const blob = await response.blob();
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play();
    """
    if not tts_server:
        raise HTTPException(status_code=503, detail="TTS server not initialized")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_data, sample_rate = tts_server.generate_audio(request)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/tts/stream")
async def stream_speech(request: TTSRequest):
    """
    Generate speech with streaming response
    Returns WAV file in chunks as it's generated

    JavaScript usage:
        const response = await fetch('http://localhost:8000/tts/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: 'Hello world', language: 'English'})
        });

        const reader = response.body.getReader();
        const chunks = [];

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunks.push(value);
        }

        const blob = new Blob(chunks, {type: 'audio/wav'});
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play();
    """
    if not tts_server:
        raise HTTPException(status_code=503, detail="TTS server not initialized")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_data, sample_rate = tts_server.generate_audio(request)

        # Generator for streaming chunks
        def generate_chunks():
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format="WAV")
            buffer.seek(0)

            # Stream in 4KB chunks
            chunk_size = 4096
            while True:
                chunk = buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        return StreamingResponse(
            generate_chunks(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def main():
    """Run the server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
