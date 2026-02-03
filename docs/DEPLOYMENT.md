# Deploying to Hugging Face Spaces with ZeroGPU

This guide explains how to deploy the Qwen3 TTS voice cloning demo to Hugging Face Spaces with ZeroGPU support.

## Prerequisites

1. **Hugging Face Account**: [Sign up](https://huggingface.co/join) if you don't have one
2. **PRO Subscription**: [Subscribe to PRO](https://huggingface.co/subscribe/pro) ($9/month) to access ZeroGPU
3. **Git LFS**: Install for handling large files
   ```bash
   git lfs install
   ```

## Deployment Steps

### 1. Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Owner**: Your username or organization
   - **Space name**: `qwen3-tts-voice-clone` (or your choice)
   - **License**: Apache 2.0
   - **SDK**: Gradio
   - **Hardware**: Select **ZeroGPU** (requires PRO subscription)
   - **Visibility**: Public or Private

### 2. Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/qwen3-tts-voice-clone
cd qwen3-tts-voice-clone
```

### 3. Copy Required Files

Copy these files from your local repo:

```bash
# Required files
cp /path/to/qwen3-tts-clone/app.py .
cp /path/to/qwen3-tts-clone/requirements.txt .
cp /path/to/qwen3-tts-clone/README_SPACES.md README.md

# Voice model
mkdir -p models
cp /path/to/qwen3-tts-clone/models/jeffrey_voice.pkl models/
```

### 4. Track Large Files with Git LFS

```bash
# Track the voice model file
git lfs track "models/*.pkl"
git add .gitattributes
```

### 5. Commit and Push

```bash
git add app.py requirements.txt README.md models/jeffrey_voice.pkl
git commit -m "feat: initial deployment of Qwen3 TTS voice clone demo"
git push
```

### 6. Wait for Build

- The Space will automatically build and deploy
- First build may take 5-10 minutes
- Check the build logs at: `https://huggingface.co/spaces/YOUR_USERNAME/qwen3-tts-voice-clone/logs`

## Configuration Details

### ZeroGPU Integration

The `app.py` file includes ZeroGPU support:

```python
import spaces

@spaces.GPU(duration=120)
def generate_speech(...):
    # GPU allocated here
    return result
    # GPU released automatically
```

**Key Points:**
- `@spaces.GPU` decorator requests GPU when function is called
- `duration=120` sets max runtime to 120 seconds
- GPU is automatically released after function completes
- Uses `torch.float16` on CUDA for efficiency

### Daily Usage Quotas

| Account Type | Daily GPU Quota | Queue Priority |
|--------------|-----------------|----------------|
| Free         | 3.5 minutes     | Medium         |
| PRO          | 25 minutes      | Highest        |
| Enterprise   | 45 minutes      | Highest        |

### Hardware Specifications

- **GPU**: Half NVIDIA H200 (default "large")
- **VRAM**: 70GB
- **Alternative**: Full H200 with `@spaces.GPU(size="xlarge")` (2× quota cost)

## Optimization Tips

### 1. Reduce Model Loading Time

Pre-load models outside the GPU-decorated function:

```python
class VoiceModelApp:
    def __init__(self):
        self.load_models()  # Load once at startup
    
    @spaces.GPU(duration=120)
    def generate_speech(self, ...):
        # Models already loaded, just generate
```

### 2. Use Float16 on GPU

```python
dtype=torch.float16 if self.device == "cuda" else torch.float32
```

### 3. Adjust Duration Based on Complexity

For shorter texts, you can reduce duration:

```python
@spaces.GPU(duration=60)  # For quick generations
```

### 4. Consider Dynamic Duration

```python
def get_duration(text, ...):
    # Estimate based on text length
    return min(120, len(text) // 10)

@spaces.GPU(duration=get_duration)
def generate_speech(self, text, ...):
    ...
```

## Troubleshooting

### Build Fails

- Check `requirements.txt` has all dependencies
- Verify `app.py` has no syntax errors
- Check build logs for specific error messages

### Out of Memory

- Reduce `max_new_tokens` default value
- Use `torch.float16` instead of `float32`
- Consider using `size="xlarge"` for more VRAM

### Slow Generation

- Check if you're hitting quota limits
- PRO users get higher queue priority
- Consider reducing `max_new_tokens` or `duration`

### Model Not Loading

- Verify `models/jeffrey_voice.pkl` is tracked with Git LFS
- Check file size limit (5GB for free, 150GB for PRO)
- Ensure file path in `app.py` matches actual path

## Alternative: Without ZeroGPU

If you prefer CPU-only (no PRO subscription needed):

1. Remove `import spaces` and `@spaces.GPU` decorator
2. Set hardware to "CPU Basic" in Space settings
3. Note: Generation will be significantly slower

## Monitoring

- **Usage**: Check your quota at https://huggingface.co/settings/billing
- **Analytics**: View Space analytics in Settings tab
- **Logs**: Monitor runtime logs for errors

## Resources

- [Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [ZeroGPU Guide](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Gradio Documentation](https://gradio.app/docs/)
- [Git LFS Guide](https://git-lfs.github.com/)
