"""
Realtime TTS using pre-created voice prompt
Fast generation without re-processing reference audio
"""

import os
import pickle
import time
from typing import Any, List

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.device_utils import (
    device_synchronize,
    get_attention_implementation,
    get_device_info,
    get_optimal_device,
    get_optimal_dtype,
)


def main() -> None:
    device = get_optimal_device()
    # Use float32 for MPS due to numerical stability issues with bfloat16
    dtype = torch.float32 if device == "mps" else get_optimal_dtype(device)
    attn_implementation = get_attention_implementation(device)

    MODEL_PATH: str = os.path.expanduser("~/LLMs/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base")
    VOICE_MODEL: str = "models/jeffrey_voice.pkl"
    OUT_DIR: str = "output"

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(VOICE_MODEL):
        print(f"Error: {VOICE_MODEL} not found")
        print("Run create_voice_prompt.py first to generate the voice model")
        return

    print(f"Device: {get_device_info(device)}")
    print(f"Dtype: {dtype}")
    print(f"Attention: {attn_implementation or 'default'}")
    print(f"\nLoading model: {MODEL_PATH}")

    model_kwargs = {
        "device_map": device,
        "dtype": dtype,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        **model_kwargs,
    )

    print("Model loaded")

    # Load pre-created voice model
    print(f"Loading voice model: {VOICE_MODEL}")
    voice_prompt: Any
    with open(VOICE_MODEL, "rb") as f:
        voice_prompt = pickle.load(f)

    print("Voice model loaded\n")

    # Test texts for realtime generation
    texts: List[str] = [
        "This is a test of realtime text to speech generation.",
        "The voice prompt is already loaded, so this should be fast.",
        "We can generate multiple samples quickly without re-processing audio.",
    ]

    gen_kwargs = {
        "max_new_tokens": 2048,
        "do_sample": True,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 0.9,
        "repetition_penalty": 1.05,
        "subtalker_dosample": True,
        "subtalker_top_k": 50,
        "subtalker_top_p": 1.0,
        "subtalker_temperature": 0.9,
        "non_streaming_mode": False,
    }

    for i, text in enumerate(texts):
        print(f"[{i + 1}/{len(texts)}] Generating: {text[:60]}...")

        device_synchronize(device)
        t0 = time.time()

        wavs, sr = tts.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=voice_prompt,
            **gen_kwargs,
        )

        device_synchronize(device)
        t1 = time.time()

        out_path = os.path.join(OUT_DIR, f"realtime_{i}.wav")
        sf.write(out_path, wavs[0], sr)
        print(f"  Generated in {t1 - t0:.2f}s -> {out_path}\n")

    print("Done. Realtime TTS complete.")


if __name__ == "__main__":
    main()
