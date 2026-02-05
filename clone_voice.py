"""
Voice cloning with Qwen3-TTS using multiple reference samples
"""

import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
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

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def main():
    device = get_optimal_device()
    # Use float32 for MPS due to numerical stability issues with bfloat16
    dtype = torch.float32 if device == "mps" else get_optimal_dtype(device)
    attn_implementation = get_attention_implementation(device)

    MODEL_PATH = os.path.expanduser("~/LLMs/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base")
    OUT_DIR = "output"
    ensure_dir(OUT_DIR)

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

    print("Model loaded\n")

    ref_paths = [
        "samples/ref_1",
        "samples/ref_2",
    ]

    print(f"Loading {len(ref_paths)} reference samples...")

    combined_audio: List[np.ndarray] = []
    combined_text: List[str] = []
    sample_rate: Optional[int] = None

    for ref_path in ref_paths:
        audio: np.ndarray
        sr: int
        audio, sr = sf.read(f"{ref_path}.wav")

        with open(f"{ref_path}.txt", "r") as f:
            text = f.read().strip()

        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise ValueError(f"Sample rate mismatch: {sample_rate} vs {sr}")

        combined_audio.append(audio)
        combined_text.append(text)
        print(f"  Loaded {ref_path}: {len(audio)} samples, {len(text)} chars")

    if sample_rate is None:
        raise ValueError("No audio files loaded")

    # Add 1 second of silence between audio files
    silence: np.ndarray = np.zeros(
        int(sample_rate * 1.0), dtype=combined_audio[0].dtype
    )
    audio_with_gaps: List[np.ndarray] = []
    for i, audio in enumerate(combined_audio):
        audio_with_gaps.append(audio)
        if i < len(combined_audio) - 1:  # Don't add silence after the last file
            audio_with_gaps.append(silence)

    concatenated_audio: np.ndarray = np.concatenate(audio_with_gaps)
    concatenated_text: str = " ".join(combined_text)

    print("\n✓ Combined into single reference:")
    print(f"  Audio: {len(concatenated_audio)} samples at {sample_rate}Hz")
    print(f"  Text: {len(concatenated_text)} characters\n")

    syn_texts = [
        "I'm not the pheasant plucker, I'm the pheasant plucker's mate. I'm only plucking pheasants 'cause the pheasant plucker's running late",
        "Extremely accurate and stunningly beautiful bespoke printer profiles transform creative print making to an extraordinary extent.",
        "Generating code from AI prompts can lead to verbose code, or duplication of existing code instead of using an abstraction. But there are times when this is perfectly acceptable, such as when building proof of concepts, or when topics like program efficiency are unimportant.",
    ]

    syn_langs = ["English"] * len(syn_texts)

    common_gen_kwargs = {
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

    print(f"Generating {len(syn_texts)} audio files...\n")

    device_synchronize(device)
    t0 = time.time()

    wavs, sr = tts.generate_voice_clone(
        text=syn_texts,
        language=syn_langs,
        ref_audio=[(concatenated_audio, sample_rate)],
        ref_text=[concatenated_text],
        x_vector_only_mode=[False],
        **common_gen_kwargs,
    )

    device_synchronize(device)
    t1 = time.time()

    print(f"\nGeneration complete: {t1 - t0:.2f}s")
    print(f"Generated {len(wavs)} files at {sr}Hz\n")

    for i, w in enumerate(wavs):
        out_path = os.path.join(OUT_DIR, f"output_{i}.wav")
        sf.write(out_path, w, sr)
        print(f"[{i + 1}/{len(wavs)}] {out_path}")

    print(f"\nDone. Files saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
