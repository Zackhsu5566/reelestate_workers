FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

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
# WanFirstLastFrameToVideo / CreateVideo / SaveVideo / QwenImageDiffsynthControlnet
# are all ComfyUI built-ins (comfy_extras/nodes_wan.py, nodes_model_patch.py)
# Only GetImageSize+ requires an external custom node
RUN cd $COMFYUI_DIR/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git comfyui_essentials && \
    pip install --no-cache-dir -r comfyui_essentials/requirements.txt || true

# Download models
RUN pip install --no-cache-dir huggingface_hub
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN \
    # UNET - high noise
    hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        split_files/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors \
        --local-dir /tmp/hf-wan && \
    mkdir -p $COMFYUI_DIR/models/diffusion_models/wan2.2 && \
    mv /tmp/hf-wan/split_files/diffusion_models/wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors \
        $COMFYUI_DIR/models/diffusion_models/wan2.2/ && \
    # UNET - low noise
    hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        split_files/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors \
        --local-dir /tmp/hf-wan && \
    mv /tmp/hf-wan/split_files/diffusion_models/wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors \
        $COMFYUI_DIR/models/diffusion_models/wan2.2/ && \
    # CLIP - text encoder
    hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
        split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
        --local-dir /tmp/hf-clip && \
    mkdir -p $COMFYUI_DIR/models/text_encoders && \
    mv /tmp/hf-clip/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
        $COMFYUI_DIR/models/text_encoders/ && \
    # VAE
    hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        split_files/vae/wan_2.1_vae.safetensors \
        --local-dir /tmp/hf-vae && \
    mkdir -p $COMFYUI_DIR/models/vae && \
    mv /tmp/hf-vae/split_files/vae/wan_2.1_vae.safetensors \
        $COMFYUI_DIR/models/vae/ && \
    # Cleanup
    rm -rf /tmp/hf-wan /tmp/hf-clip /tmp/hf-vae

# Copy handler and workflows
COPY handler.py /handler.py
COPY workflows/ /workflows/
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
