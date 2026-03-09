"""
Worker B: Wan 2.2 — first-last-frame to video generation
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


def wait_for_completion(prompt_id: str, timeout: int = 600) -> dict:
    """Longer timeout for Wan 2.2 (can take 4-8 min)."""
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
        time.sleep(5)
    raise TimeoutError(f"Workflow {prompt_id} timed out after {timeout}s")


def load_workflow(name: str) -> dict:
    with open(f"/workflows/{name}", "r", encoding="utf-8") as f:
        return json.load(f)


def handler(job):
    """
    Input:
    {
        "job_id": "xxx",
        "first_frame_url": "https://r2.../first.jpg",
        "last_frame_url": "https://r2.../last.jpg",
        "clip_name": "主臥"
    }

    Output:
    {
        "video_url": "https://r2.../job_id/clips/主臥.mp4",
        "clip_name": "主臥"
    }
    """
    job_input = job["input"]
    job_id = job_input.get("job_id", str(uuid.uuid4()))
    first_url = job_input["first_frame_url"]
    last_url = job_input["last_frame_url"]
    clip_name = job_input["clip_name"]

    # Download frames
    download_file(first_url, "/comfyui/input/first.jpg")
    download_file(last_url, "/comfyui/input/last.jpg")

    # Load workflow and inject parameters
    workflow = load_workflow("_Wan2.2_fun_camera_FLF2V.json")

    # Node 15 (LoadImage): first frame
    workflow["15"]["inputs"]["image"] = "first.jpg"
    # Node 16 (LoadImage): last frame
    workflow["16"]["inputs"]["image"] = "last.jpg"
    # Node 17 (WanFirstLastFrameToVideo): dimensions and length
    width = job_input.get("width", 512)
    height = job_input.get("height", 512)
    length = job_input.get("num_frames", 81)
    workflow["17"]["inputs"]["width"] = width
    workflow["17"]["inputs"]["height"] = height
    workflow["17"]["inputs"]["length"] = length
    # Node 10 (KSamplerAdvanced): randomize seed
    import random
    workflow["10"]["inputs"]["noise_seed"] = random.randint(0, 2**53)

    prompt_id = queue_workflow(workflow)
    outputs = wait_for_completion(prompt_id)

    # Node 9 (SaveVideo) is the output
    videos = outputs.get("9", {}).get("videos", outputs.get("9", {}).get("gifs", []))
    if not videos:
        raise RuntimeError("Wan 2.2 workflow did not produce video output")

    vid_info = videos[0]
    subfolder = vid_info.get("subfolder", "")
    filename = vid_info["filename"]
    video_path = f"/comfyui/output/{subfolder}/{filename}" if subfolder else f"/comfyui/output/{filename}"

    # Upload to R2
    r2_key = f"{job_id}/clips/{clip_name}.mp4"
    video_url = upload_to_r2(video_path, r2_key)

    return {
        "video_url": video_url,
        "clip_name": clip_name,
    }


runpod.serverless.start({"handler": handler})
