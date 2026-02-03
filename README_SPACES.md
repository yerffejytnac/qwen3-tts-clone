---
title: Qwen3 TTS Voice Clone
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Qwen3 TTS Voice Clone Demo

Voice cloning demo using Qwen3-TTS-12Hz-1.7B-Base with custom voice model.

This Space demonstrates voice cloning capabilities with:
- Multi-language support (English, Chinese, Japanese, Korean, French, German, Spanish)
- Advanced sampling parameters for quality control
- Subtalker controls for improved synthesis
- Real-time generation using pre-trained voice model

## Usage

1. Enter text to synthesize
2. Select language
3. (Optional) Adjust advanced parameters
4. Click "Generate Speech"

## Model

- Base Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Custom Voice: Pre-trained on reference audio samples
- Generation: ~120s per synthesis on ZeroGPU

## Links

- [GitHub Repository](https://github.com/yerffejytnac/qwen3-tts-clone)
- [Qwen3-TTS Model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
