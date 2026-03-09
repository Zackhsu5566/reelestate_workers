#!/bin/bash
set -e

# 1. Start ComfyUI in background
echo "Starting ComfyUI..."
cd /comfyui && python main.py --listen 127.0.0.1 --port 8188 &

# 2. Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
while ! curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; do
    sleep 1
done
echo "ComfyUI is ready."

# 3. Start RunPod handler
echo "Starting RunPod handler..."
python /handler.py
