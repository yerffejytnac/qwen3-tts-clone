# Running TTS API Locally on Mac

Simple guide for running the TTS API server on your Mac with MPS acceleration.

## Quick Start

### 1. Start the API Server

```bash
./start_api.sh
```

Or manually:

```bash
uv run python api_server.py
```

The server will start at `http://localhost:8000` and automatically use MPS (Apple Silicon GPU).

### 2. Test It

**Option A: Web UI**
- Open `demo.html` in your browser
- Type text and click "Generate Speech"

**Option B: Command Line**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from my Mac","language":"English"}' \
  --output test.wav && open test.wav
```

## Integration Examples

### HTML/JavaScript (Local Web App)

```html
<!DOCTYPE html>
<html>
<body>
  <textarea id="text">Hello world</textarea>
  <button onclick="speak()">Speak</button>
  <audio id="player" controls></audio>

  <script>
    async function speak() {
      const text = document.getElementById('text').value;
      
      const response = await fetch('http://localhost:8000/tts', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text, language: 'English'})
      });
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      document.getElementById('player').src = url;
      document.getElementById('player').play();
    }
  </script>
</body>
</html>
```

### Electron App

```javascript
// In renderer process
async function generateSpeech(text) {
  const response = await fetch('http://localhost:8000/tts', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, language: 'English'})
  });
  
  const arrayBuffer = await response.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  source.start(0);
}
```

### AppleScript (System Integration)

```applescript
-- Call TTS API from AppleScript
set theText to "Hello from AppleScript"
set jsonData to "{\"text\":\"" & theText & "\",\"language\":\"English\"}"

do shell script "curl -X POST http://localhost:8000/tts " & ¬
  "-H 'Content-Type: application/json' " & ¬
  "-d '" & jsonData & "' " & ¬
  "--output /tmp/speech.wav && open /tmp/speech.wav"
```

### Raycast Extension

```typescript
// Raycast script command
import { showToast, Toast } from "@raycast/api";

export default async function Command(props: { arguments: { text: string } }) {
  const { text } = props.arguments;
  
  showToast({
    style: Toast.Style.Animated,
    title: "Generating speech...",
  });
  
  const response = await fetch('http://localhost:8000/tts', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, language: 'English'})
  });
  
  const blob = await response.blob();
  const arrayBuffer = await blob.arrayBuffer();
  
  // Save and play
  const fs = require('fs');
  fs.writeFileSync('/tmp/raycast-tts.wav', Buffer.from(arrayBuffer));
  
  const { exec } = require('child_process');
  exec('afplay /tmp/raycast-tts.wav');
  
  showToast({
    style: Toast.Style.Success,
    title: "Playing speech",
  });
}
```

### Alfred Workflow

```bash
# Alfred Script Filter
query="{query}"

curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$query\",\"language\":\"English\"}" \
  --output /tmp/alfred-tts.wav

afplay /tmp/alfred-tts.wav
```

### Python Script

```python
import requests
import subprocess

def speak(text: str):
    response = requests.post(
        'http://localhost:8000/tts',
        json={'text': text, 'language': 'English'}
    )
    
    with open('/tmp/tts.wav', 'wb') as f:
        f.write(response.content)
    
    # Play with afplay (macOS built-in)
    subprocess.run(['afplay', '/tmp/tts.wav'])

speak("Hello from Python")
```

## Performance on Mac

### Apple Silicon (M1/M2/M3)
- Device: **MPS** (Metal Performance Shaders)
- Speed: ~10-15s for typical sentence
- Memory: ~4GB VRAM

### Intel Mac
- Device: **CPU**
- Speed: ~30-60s for typical sentence
- Memory: ~4GB RAM

## Optimizations for Mac

### 1. Faster Generation (Lower Quality)

```javascript
fetch('http://localhost:8000/tts', {
  method: 'POST',
  body: JSON.stringify({
    text: 'Hello',
    x_vector_only: true  // 2-3x faster
  })
})
```

### 2. Adjust Token Length

```javascript
{
  text: 'Short text',
  max_new_tokens: 1024  // Default: 2048 (less = faster)
}
```

### 3. Batch Requests

```javascript
// Generate multiple at once
const texts = ['Hello', 'Goodbye', 'Thank you'];

const responses = await Promise.all(
  texts.map(text => 
    fetch('http://localhost:8000/tts', {
      method: 'POST',
      body: JSON.stringify({text})
    })
  )
);
```

## Autostart with launchd (Optional)

Create `~/Library/LaunchAgents/com.tts.api.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tts.api</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/Development/qwen3-tts-clone/start_api.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/tts-api.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/tts-api-error.log</string>
</dict>
</plist>
```

Then:
```bash
launchctl load ~/Library/LaunchAgents/com.tts.api.plist
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>
```

### MPS Not Available

If MPS isn't detected, the server will fall back to CPU automatically.

Check with:
```python
import torch
print(torch.backends.mps.is_available())
```

### Audio Not Playing

Make sure you have audio output selected:
```bash
# List audio devices
system_profiler SPAudioDataType

# Test audio
say "testing audio"
```

## Monitoring

### Check Server Status

```bash
curl http://localhost:8000/health
```

### View Logs

Server logs print to stdout. For background running:
```bash
./start_api.sh > /tmp/tts-api.log 2>&1 &
tail -f /tmp/tts-api.log
```

### Activity Monitor

Watch GPU usage in Activity Monitor > Window > GPU History

## Stopping the Server

Press `Ctrl+C` in the terminal, or:

```bash
pkill -f api_server.py
```
