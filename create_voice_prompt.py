"""
Create a reusable voice clone prompt from reference samples
This prompt can be saved and reused for realtime TTS without re-processing
"""

import os
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

AudioLike = Union[str, np.ndarray, Tuple[np.ndarray, int]]


def main() -> None:
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    MODEL_PATH: str = os.path.expanduser("~/LLMs/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base")

    print(f"Device: {device}")
    print(f"Loading model: {MODEL_PATH}")

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.float32,
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
    print(f"  Text: {len(concatenated_text)} characters")

    print("\nCreating voice prompt from combined reference...")

    # Create reusable voice clone prompt with single concatenated reference
    voice_prompt = tts.create_voice_clone_prompt(
        ref_audio=(concatenated_audio, sample_rate),
        ref_text=concatenated_text,
        x_vector_only_mode=False,
    )

    print("✓ Voice prompt created")
    print(f"Type: {type(voice_prompt)}")
    print(f"Items: {len(voice_prompt)}")

    # Save to models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    voice_model_file = os.path.join(models_dir, "jeffrey_voice.pkl")
    with open(voice_model_file, "wb") as f:
        pickle.dump(voice_prompt, f)

    print(f"\n✓ Voice model saved to: {voice_model_file}")
    print("This voice model can be reused for realtime TTS without re-processing audio")


if __name__ == "__main__":
    main()
