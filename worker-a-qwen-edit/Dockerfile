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

# Install RunPod SDK + boto3 (R2 upload)
RUN pip install --no-cache-dir runpod boto3 requests

# Custom Nodes
RUN cd $COMFYUI_DIR/custom_nodes && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git && \
    pip install --no-cache-dir -r ComfyUI-Easy-Use/requirements.txt || true

# Download models
RUN pip install --no-cache-dir huggingface_hub && \
    # UNET
    huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI \
        split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors \
        --local-dir /tmp/hf && \
    mkdir -p $COMFYUI_DIR/models/diffusion_models/Qwen && \
    mv /tmp/hf/split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors \
        $COMFYUI_DIR/models/diffusion_models/Qwen/ && \
    # CLIP
    huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI \
        split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
        --local-dir /tmp/hf && \
    mkdir -p $COMFYUI_DIR/models/clip && \
    mv /tmp/hf/split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
        $COMFYUI_DIR/models/clip/ && \
    # VAE
    huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI \
        split_files/vae/qwen_image_vae.safetensors \
        --local-dir /tmp/hf && \
    mkdir -p $COMFYUI_DIR/models/vae && \
    mv /tmp/hf/split_files/vae/qwen_image_vae.safetensors \
        $COMFYUI_DIR/models/vae/ && \
    # LoRA - Camera angle
    huggingface-cli download dx8152/Qwen-Edit-2509-Multiple-angles \
        镜头转换.safetensors \
        --local-dir /tmp/hf-lora && \
    mkdir -p $COMFYUI_DIR/models/loras/Qwen && \
    mv "/tmp/hf-lora/镜头转换.safetensors" \
        $COMFYUI_DIR/models/loras/Qwen/ && \
    # LoRA - Lightning acceleration
    huggingface-cli download lightx2v/Qwen-Image-Lightning \
        Qwen-Image-Lightning-8steps-V2.0.safetensors \
        --local-dir /tmp/hf-lightning && \
    mv /tmp/hf-lightning/Qwen-Image-Lightning-8steps-V2.0.safetensors \
        $COMFYUI_DIR/models/loras/Qwen/ && \
    # Cleanup
    rm -rf /tmp/hf /tmp/hf-lora /tmp/hf-lightning

# Copy handler and workflows
COPY handler.py /handler.py
COPY workflows/ /workflows/
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
