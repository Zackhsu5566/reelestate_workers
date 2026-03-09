FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV COMFYUI_DIR=/comfyui

# System deps (ffmpeg needed for video processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_DIR && \
    pip install --no-cache-dir -r $COMFYUI_DIR/requirements.txt

# Install RunPod SDK + boto3 (R2 upload)
RUN pip install --no-cache-dir runpod boto3 requests

# Custom Nodes
RUN cd $COMFYUI_DIR/custom_nodes && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    pip install --no-cache-dir -r comfyui_controlnet_aux/requirements.txt || true && \
    git clone https://github.com/rgthree/rgthree-comfy.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt || true && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    pip install --no-cache-dir -r ComfyUI-Frame-Interpolation/requirements.txt || true && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git && \
    pip install --no-cache-dir -r ComfyUI-Easy-Use/requirements.txt || true && \
    git clone https://github.com/danTheMonk/comfyui-int-and-float.git

# Download models -- Z-Image
RUN pip install --no-cache-dir huggingface_hub && \
    # UNET
    huggingface-cli download Comfy-Org/z_image_turbo \
        split_files/diffusion_models/z_image_turbo_bf16.safetensors \
        --local-dir /tmp/hf-zi && \
    mkdir -p $COMFYUI_DIR/models/diffusion_models && \
    mv /tmp/hf-zi/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
        $COMFYUI_DIR/models/diffusion_models/ && \
    # CLIP
    huggingface-cli download Comfy-Org/z_image_turbo \
        split_files/text_encoders/qwen_3_4b.safetensors \
        --local-dir /tmp/hf-zi && \
    mkdir -p $COMFYUI_DIR/models/text_encoders && \
    mv /tmp/hf-zi/split_files/text_encoders/qwen_3_4b.safetensors \
        $COMFYUI_DIR/models/text_encoders/ && \
    # VAE
    huggingface-cli download Comfy-Org/z_image_turbo \
        split_files/vae/ae.safetensors \
        --local-dir /tmp/hf-zi && \
    mkdir -p $COMFYUI_DIR/models/vae && \
    mv /tmp/hf-zi/split_files/vae/ae.safetensors \
        $COMFYUI_DIR/models/vae/ && \
    # ControlNet
    huggingface-cli download alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union \
        Z-Image-Turbo-Fun-Controlnet-Union.safetensors \
        --local-dir /tmp/hf-cn && \
    mkdir -p $COMFYUI_DIR/models/model_patches && \
    mv /tmp/hf-cn/Z-Image-Turbo-Fun-Controlnet-Union.safetensors \
        $COMFYUI_DIR/models/model_patches/ && \
    # Cleanup
    rm -rf /tmp/hf-zi /tmp/hf-cn

# Download models -- Upscale
RUN curl -L -o $COMFYUI_DIR/models/upscale_models/4x_NMKD-Siax_200k.pth \
    "https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth" || \
    echo "WARNING: Upscale model download failed, please verify URL"
# rife47.pth is auto-downloaded by ComfyUI-Frame-Interpolation

# Copy handler and workflows
COPY handler.py /handler.py
COPY workflows/ /workflows/
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
