# Realtime TTS

## Overview

Instead of re-processing reference audio every time, create a reusable voice prompt once and use it for fast realtime TTS.

## Workflow

### 1. Create Voice Prompt (One-Time)

```bash
uv run python create_voice_prompt.py
```

This processes your reference samples and saves a voice prompt file (`voice_prompt.pkl`).

**Output:**
- `models/jeffrey_voice.pkl` - Reusable voice model combining both reference samples

### 2. Realtime Generation

```bash
uv run python realtime_tts.py
```

Uses the pre-created prompt for fast generation without re-processing audio.

## Benefits

- **No Re-Processing**: Reference audio processed once
- **Faster Generation**: Skip audio encoding on each request
- **Consistent Voice**: Same prompt ensures voice consistency
- **Realtime Ready**: Suitable for interactive applications

## Technical Details

The voice prompt contains:
- Speaker embeddings (`ref_spk_embedding`)
- Encoded reference codes (`ref_code`)
- Reference text for ICL mode
- Mode flags (`x_vector_only_mode`, `icl_mode`)

## Integration

For web apps or APIs, load the voice model once on startup:

```python
# Load once
with open("models/jeffrey_voice.pkl", "rb") as f:
    voice_model = pickle.load(f)

# Reuse for all requests (batch mode with multiple prompts)
def generate_speech(texts):
    wavs, sr = tts.generate_voice_clone(
        text=texts if isinstance(texts, list) else [texts],
        language=["English"] * (len(texts) if isinstance(texts, list) else 1),
        voice_clone_prompt=voice_model,
    )
    return wavs, sr
```

**Note:** The voice model contains a single concatenated prompt from all reference samples.

## Re-Creating Prompt

If you update reference samples, run `create_voice_prompt.py` again to regenerate the prompt.
