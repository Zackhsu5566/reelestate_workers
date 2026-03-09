"""
Worker D: Z-Image virtual staging + Video upscale
Actions: "staging" | "upscale"
"""

import os
import json
import time
import uuid
import requests
import runpod
import boto3

# ── R2 config ──
R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "")
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_BUCKET = os.environ.get("R2_BUCKET", "reelestate-assets")

COMFYUI_URL = "http://127.0.0.1:8188"


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
    )


def upload_to_r2(local_path: str, r2_key: str) -> str:
    s3 = get_s3_client()
    s3.upload_file(local_path, R2_BUCKET, r2_key)
    return f"{R2_ENDPOINT}/{R2_BUCKET}/{r2_key}"


def download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def queue_workflow(workflow: dict) -> str:
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        data = resp.json()
        if prompt_id in data:
            status = data[prompt_id].get("status", {})
            if status.get("completed", False) or status.get("status_str") == "success":
                return data[prompt_id]["outputs"]
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI workflow error: {status}")
        time.sleep(2)
    raise TimeoutError(f"Workflow {prompt_id} timed out after {timeout}s")


def load_workflow(name: str) -> dict:
    with open(f"/workflows/{name}", "r", encoding="utf-8") as f:
        return json.load(f)


def action_staging(job_input: dict) -> dict:
    """
    Z-Image virtual staging.

    Input:
    {
        "task_type": "staging",
        "job_id": "xxx",
        "image_url": "https://r2.../客廳.jpg",
        "space_name": "客廳",
        "prompt": "modern minimalist living room..."
    }
    """
    job_id = job_input.get("job_id", str(uuid.uuid4()))
    image_url = job_input["image_url"]
    space_name = job_input["space_name"]
    prompt = job_input["prompt"]

    # Download source image
    download_file(image_url, "/comfyui/input/source.jpg")
    negative_prompt = job_input.get("negative_prompt",
        "cgi, render, 3d render, unreal engine, perfect lighting, studio lighting, "
        "overly clean, fake shadows, floating furniture, distorted geometry, "
        "unrealistic perspective, cartoon, illustration")

    workflow = load_workflow("z-image-staging.json")

    # Node 58 (LoadImage): source image
    workflow["58"]["inputs"]["image"] = "source.jpg"
    # Node 88 (CLIPTextEncode): positive prompt
    workflow["88"]["inputs"]["text"] = prompt
    # Node 90 (CLIPTextEncode): negative prompt
    workflow["90"]["inputs"]["text"] = negative_prompt
    # Node 83 (KSampler): randomize seed
    import random
    workflow["83"]["inputs"]["seed"] = random.randint(0, 2**53)

    prompt_id = queue_workflow(workflow)
    outputs = wait_for_completion(prompt_id)

    # Node 73 (PreviewImage) is the output
    images = outputs.get("73", {}).get("images", [])
    if not images:
        raise RuntimeError("Z-Image workflow did not produce image output")

    img_info = images[0]
    subfolder = img_info.get("subfolder", "")
    filename = img_info["filename"]
    output_path = f"/comfyui/output/{subfolder}/{filename}" if subfolder else f"/comfyui/output/{filename}"

    r2_key = f"{job_id}/images/{space_name}.jpg"
    staged_url = upload_to_r2(output_path, r2_key)

    return {
        "staged_image_url": staged_url,
        "space_name": space_name,
    }


def action_upscale(job_input: dict) -> dict:
    """
    Video upscale with frame interpolation.

    Input:
    {
        "task_type": "upscale",
        "job_id": "xxx",
        "video_url": "https://r2.../clips/客廳1.mp4",
        "clip_name": "客廳1"
    }
    """
    job_id = job_input.get("job_id", str(uuid.uuid4()))
    video_url = job_input["video_url"]
    clip_name = job_input["clip_name"]

    # Download video
    download_file(video_url, "/comfyui/input/clip.mp4")

    workflow = load_workflow("Video Diffusion Upscaler!.json")

    # Node 1 (VHS_LoadVideo): input video
    workflow["1"]["inputs"]["video"] = "clip.mp4"

    prompt_id = queue_workflow(workflow)
    outputs = wait_for_completion(prompt_id, timeout=600)

    # Node 44 (VHS_VideoCombine) is the upscaled + interpolated output
    videos = outputs.get("44", {}).get("gifs", outputs.get("44", {}).get("videos", []))
    if not videos:
        # Fallback to node 8 (upscale only, no interpolation)
        videos = outputs.get("8", {}).get("gifs", outputs.get("8", {}).get("videos", []))
    if not videos:
        raise RuntimeError("Upscale workflow did not produce video output")

    vid_info = videos[0]
    subfolder = vid_info.get("subfolder", "")
    filename = vid_info["filename"]
    video_path = f"/comfyui/output/{subfolder}/{filename}" if subfolder else f"/comfyui/output/{filename}"

    r2_key = f"{job_id}/clips_hq/{clip_name}.mp4"
    upscaled_url = upload_to_r2(video_path, r2_key)

    return {
        "upscaled_url": upscaled_url,
        "clip_name": clip_name,
    }


def handler(job):
    job_input = job["input"]
    task_type = job_input.get("task_type")

    if task_type == "staging":
        return action_staging(job_input)
    elif task_type == "upscale":
        return action_upscale(job_input)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Expected 'staging' or 'upscale'.")


runpod.serverless.start({"handler": handler})
