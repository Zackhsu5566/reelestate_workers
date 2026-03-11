"""
Worker A: Qwen Image Edit — second angle generation
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
R2_CDN = os.environ.get("R2_CDN", "https://assets.replowapp.com")

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
    return f"{R2_CDN}/{r2_key}"


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


def handler(job):
    """
    Input:
    {
        "job_id": "xxx",
        "image_url": "https://r2.../photo.jpg",
        "image_name": "bedroom.jpg",
        "space_name": "主臥",
        "prompt": "将镜头向左旋转35度"
    }

    Output:
    {
        "generated_url": "https://r2.../job_id/angles/主臥_angle2.jpg",
        "space_name": "主臥"
    }
    """
    job_input = job["input"]
    job_id = job_input.get("job_id", str(uuid.uuid4()))
    image_url = job_input["image_url"]
    image_name = job_input.get("image_name", "input.jpg")
    space_name = job_input["space_name"]
    prompt = job_input["prompt"]

    # Download source image
    local_image = f"/comfyui/input/{image_name}"
    download_file(image_url, local_image)

    # Load workflow and inject parameters
    workflow = load_workflow("QwenImageEdit2509Cameracontrol.json")

    # Node 31 (LoadImage): source image
    workflow["31"]["inputs"]["image"] = image_name
    # Node 85 (CR Text): camera direction prompt
    workflow["85"]["inputs"]["text"] = prompt
    # Node 14 (KSampler): randomize seed
    import random
    workflow["14"]["inputs"]["seed"] = random.randint(0, 2**53)

    prompt_id = queue_workflow(workflow)
    outputs = wait_for_completion(prompt_id)

    # Node 91 (SaveImage) is the output
    images = outputs.get("91", {}).get("images", [])
    if not images:
        raise RuntimeError("Qwen Edit workflow did not produce image output")

    img_info = images[0]
    subfolder = img_info.get("subfolder", "")
    filename = img_info["filename"]
    output_path = f"/comfyui/output/{subfolder}/{filename}" if subfolder else f"/comfyui/output/{filename}"

    # Upload to R2
    r2_key = f"{job_id}/angles/{space_name}_angle2.jpg"
    generated_url = upload_to_r2(output_path, r2_key)

    return {
        "generated_url": generated_url,
        "space_name": space_name,
    }


runpod.serverless.start({"handler": handler})
