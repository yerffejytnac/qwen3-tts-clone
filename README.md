# Qwen3-TTS Voice Cloning

Voice cloning using Qwen3-TTS-12Hz-1.7B-Base with combined reference samples.

## Setup

```bash
uv init --python 3.12
uv add qwen-tts soundfile torch numpy
```

## Usage

```bash
uv run python clone_voice.py
```

Generates speech using 2 combined reference samples for improved voice quality.

**Performance:** ~111s for 3 outputs on MPS (Apple Silicon)

## Reference Format

Reference files follow the pattern: `samples/ref_N` (without extension)
- `ref_1.wav` + `ref_1.txt`
- `ref_2.wav` + `ref_2.txt`

Add or remove references by editing `ref_paths` list in `clone_voice.py`. References are concatenated with 1s silence gaps.

## Output

Generated audio samples:

| File | Phrase |
|------|--------|
| [output_0.wav](output/output_0.wav) | I'm not the pheasant plucker, I'm the pheasant plucker's mate. I'm only plucking pheasants 'cause the pheasant plucker's running late |
| [output_1.wav](output/output_1.wav) | Extremely accurate and stunningly beautiful bespoke printer profiles transform creative print making to an extraordinary extent. |
| [output_2.wav](output/output_2.wav) | Generating code from AI prompts can lead to verbose code, or duplication of existing code instead of using an abstraction. But there are times when this is perfectly acceptable, such as when building proof of concepts, or when topics like program efficiency are unimportant. |

## Realtime TTS

### Command Line

For fast realtime generation without re-processing reference audio:

```bash
# 1. Create voice model once
uv run python create_voice_prompt.py

# 2. Use for realtime generation
uv run python realtime_tts.py
```

### Web UI Demo

Launch the Gradio web interface:

```bash
uv run python demo_app.py
```

Then open `http://localhost:8000` in your browser.

See [docs/realtime.md](docs/realtime.md) for details.

## Configuration

See [docs/configuration.md](docs/configuration.md) for generation parameters.

Edit `syn_texts` in `clone_voice.py` to customize synthesis text.

## Model

Location: `~/LLMs/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base`

Device: Auto-detects MPS or CPU

## References

- [Qwen3-TTS Model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Example Code](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py)
