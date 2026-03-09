FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV COMFYUI_DIR=/comfyui

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_DIR && \
    pip install --no-cache-dir -r $COMFYUI_DIR/requirements.txt

# Install RunPod SDK + boto3 (R2 upload) + jieba (post-processing)
RUN pip install --no-cache-dir runpod boto3 requests jieba

# Custom Nodes
RUN cd $COMFYUI_DIR/custom_nodes && \
    git clone https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git && \
    pip install --no-cache-dir -r ComfyUI-Qwen-TTS/requirements.txt || true && \
    git clone https://github.com/GentlemanHu/comfyui-edgetts.git && \
    pip install --no-cache-dir -r comfyui-edgetts/requirements.txt || true && \
    git clone https://github.com/kaushiknishchay/ComfyUI-Qwen3-ASR.git && \
    pip install --no-cache-dir -r ComfyUI-Qwen3-ASR/requirements.txt || true

# Models are auto-downloaded by custom nodes on first run:
# - Qwen3-TTS 1.7B
# - Whisper base
# - Qwen3-ASR 1.7B
# - Qwen3-ForcedAligner 0.6B

# Copy handler, post-processing script, and workflows
COPY handler.py /handler.py
COPY process_alignment.py /process_alignment.py
COPY workflows/ /workflows/
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
