# TTS API Documentation

RESTful API for text-to-speech generation with streaming support.

## Quick Start

### 1. Start the Server

```bash
uv run python api_server.py
```

Server will start at `http://localhost:8000`

### 2. Test with JavaScript

Open `demo.html` in your browser or use the examples below.

## Endpoints

### GET `/`

Get API information and available endpoints.

**Response:**
```json
{
  "name": "Qwen3-TTS API",
  "version": "1.0.0",
  "endpoints": {
    "POST /tts": "Generate speech from text (returns WAV file)",
    "POST /tts/stream": "Generate speech with streaming",
    "GET /health": "Health check"
  }
}
```

### GET `/health`

Check server status.

**Response:**
```json
{
  "status": "ok",
  "device": "mps"
}
```

### POST `/tts`

Generate speech and return complete WAV file.

**Request Body:**
```json
{
  "text": "Hello world",
  "language": "English",
  "x_vector_only": false,
  "max_new_tokens": 2048,
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "repetition_penalty": 1.05,
  "subtalker_temperature": 0.9,
  "subtalker_top_p": 1.0,
  "subtalker_top_k": 50
}
```

**Parameters:**
- `text` (required): Text to synthesize
- `language` (optional): Language (default: "English")
  - Options: English, Chinese, Japanese, Korean, French, German, Spanish
- `x_vector_only` (optional): Faster mode with lower quality (default: false)
- `max_new_tokens` (optional): Maximum output length (default: 2048)
- `temperature` (optional): Randomness 0.1-1.5 (default: 0.9)
- `top_p` (optional): Nucleus sampling (default: 1.0)
- `top_k` (optional): Sampling diversity (default: 50)
- `repetition_penalty` (optional): Reduce repetition (default: 1.05)
- `subtalker_*`: Fine-grained voice control parameters

**Response:**
- Content-Type: `audio/wav`
- Binary WAV file

### POST `/tts/stream`

Generate speech with streaming response (chunks as generated).

**Request Body:** Same as `/tts`

**Response:**
- Content-Type: `audio/wav`
- Chunked WAV file

## JavaScript Examples

### Basic Usage

```javascript
const response = await fetch('http://localhost:8000/tts', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Hello world',
    language: 'English'
  })
});

const blob = await response.blob();
const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

### Streaming with Progress

```javascript
const response = await fetch('http://localhost:8000/tts/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Hello world',
    language: 'English'
  })
});

const reader = response.body.getReader();
const chunks = [];
let receivedLength = 0;

while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  
  chunks.push(value);
  receivedLength += value.length;
  console.log(`Received ${receivedLength} bytes`);
}

const blob = new Blob(chunks, {type: 'audio/wav'});
const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

### Advanced Parameters

```javascript
const response = await fetch('http://localhost:8000/tts', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Hello world',
    language: 'English',
    temperature: 0.7,      // Less random
    top_k: 30,             // More focused
    x_vector_only: true    // Faster mode
  })
});
```

## Python Examples

### Using `requests`

```python
import requests

response = requests.post(
    'http://localhost:8000/tts',
    json={
        'text': 'Hello world',
        'language': 'English'
    }
)

with open('output.wav', 'wb') as f:
    f.write(response.content)
```

### Using `httpx` with Streaming

```python
import httpx

with httpx.stream(
    'POST',
    'http://localhost:8000/tts/stream',
    json={'text': 'Hello world', 'language': 'English'}
) as response:
    with open('output.wav', 'wb') as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
            print(f"Received {len(chunk)} bytes")
```

## cURL Examples

### Basic Request

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"English"}' \
  --output speech.wav
```

### Streaming Request

```bash
curl -X POST http://localhost:8000/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"English"}' \
  --output speech.wav
```

## Error Handling

### JavaScript

```javascript
try {
  const response = await fetch('http://localhost:8000/tts', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: 'Hello world'})
  });
  
  if (!response.ok) {
    const error = await response.json();
    console.error('Error:', error.detail);
    return;
  }
  
  const blob = await response.blob();
  const audio = new Audio(URL.createObjectURL(blob));
  audio.play();
  
} catch (error) {
  console.error('Network error:', error);
}
```

## CORS

The API enables CORS for all origins by default. For production, update `allow_origins` in `api_server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## Production Deployment

### Using Gunicorn (multiple workers)

```bash
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

EXPOSE 8000
CMD ["uv", "run", "python", "api_server.py"]
```

### Environment Variables

```bash
export TTS_MODEL_PATH="~/LLMs/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-Base"
export VOICE_MODEL_PATH="models/jeffrey_voice.pkl"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

## Performance Tips

1. **Use `x_vector_only: true`** for faster generation (lower quality)
2. **Reduce `max_new_tokens`** for shorter responses
3. **Lower `temperature`** for more consistent output
4. **Stream large responses** to start playback sooner
5. **Deploy on GPU** (CUDA/MPS) for 10-30x speedup

## Monitoring

### Check Server Status

```bash
curl http://localhost:8000/health
```

### View Logs

The server outputs generation timing and errors to stdout.
