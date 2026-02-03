# Configuration

## Device

Auto-detects MPS (Apple Silicon) or CPU:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

## Model Loading

```python
tts = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map=device,
    dtype=torch.float32,
)
```

## Reference Files

Pattern: `samples/ref_N` (without extension)

```python
ref_paths = [
    "samples/ref_1",
    "samples/ref_2",
]
```

Each reference needs `.wav` and `.txt` files. References are concatenated with 1s silence gaps.

## Generation Parameters

```python
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
```

### Key Parameters

- **max_new_tokens**: Maximum output length
- **temperature**: Randomness (0.1-1.5, higher = more varied)
- **top_k/top_p**: Sampling diversity controls
- **repetition_penalty**: Reduces repetition (>1.0)
- **subtalker_***: Subtalker-specific sampling parameters for improved quality
- **x_vector_only_mode**: `False` for ICL (better quality), `True` for speed

## Synthesis Configuration

```python
syn_texts = [
    "Your text here.",
    "Another sentence.",
]

syn_langs = ["English"] * len(syn_texts)
```

## Output

Files saved as `output/output_N.wav` matching `syn_texts` order.
