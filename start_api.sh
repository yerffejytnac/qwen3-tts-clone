#!/bin/bash
# Start the TTS API server locally on Mac

echo "🎙️  Starting Qwen3-TTS API Server..."
echo ""
echo "Server will be available at:"
echo "  http://localhost:8000"
echo ""
echo "Open demo.html in your browser to test"
echo "Press Ctrl+C to stop"
echo ""

# Start the server
uv run python api_server.py
