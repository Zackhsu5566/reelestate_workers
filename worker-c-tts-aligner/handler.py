"""
Worker C: TTS + ForcedAligner + post-processing
Actions: "tts" | "align"
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
    """Upload file to R2 and return public URL."""
    s3 = get_s3_client()
    s3.upload_file(local_path, R2_BUCKET, r2_key)
    return f"{R2_ENDPOINT}/{R2_BUCKET}/{r2_key}"


def download_file(url: str, dest: str):
    """Download a file from URL to local path."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def queue_workflow(workflow: dict) -> str:
    """Submit workflow to ComfyUI and return prompt_id."""
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Poll ComfyUI until workflow completes. Returns output dict."""
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
    """Load a workflow JSON from /workflows/ directory."""
    path = f"/workflows/{name}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Actions ──

def action_tts(job_input: dict) -> dict:
    """Run TTS and return audio URL."""
    script = job_input["script"]
    job_id = job_input.get("job_id", str(uuid.uuid4()))
    voice = job_input.get("voice", {})

    # TTS needs a reference audio for voice cloning
    # Agent should provide ref_audio_url + ref_text, or use defaults
    ref_audio_url = job_input.get("ref_audio_url")
    ref_text = job_input.get("ref_text", "")

    # Download reference audio if provided
    if ref_audio_url:
        download_file(ref_audio_url, "/comfyui/input/ref_audio.mp3")
        ref_audio_name = "ref_audio.mp3"
    else:
        ref_audio_name = ""

    workflow = load_workflow("Qwen3-TTS.json")

    # Node 48 (FB_Qwen3TTSVoiceClone): TTS parameters
    workflow["48"]["inputs"]["target_text"] = script
    workflow["48"]["inputs"]["ref_text"] = ref_text
    workflow["48"]["inputs"]["model_choice"] = voice.get("model_choice", "1.7B")
    workflow["48"]["inputs"]["language"] = voice.get("language", "Chinese")
    workflow["48"]["inputs"]["top_p"] = voice.get("top_p", 0.8)
    workflow["48"]["inputs"]["temperature"] = voice.get("temperature", 1)
    import random
    workflow["48"]["inputs"]["seed"] = random.randint(0, 2**53)

    # Node 24 (LoadAudio): reference audio for voice clone
    if ref_audio_name:
        workflow["24"]["inputs"]["audio"] = ref_audio_name

    prompt_id = queue_workflow(workflow)
    outputs = wait_for_completion(prompt_id, timeout=180)

    # Node 31 (PreviewAudio) is the output
    audio_path = None
    for node_id, node_output in outputs.items():
        if "audio" in node_output:
            audio_file = node_output["audio"][0]
            subfolder = audio_file.get("subfolder", "")
            filename = audio_file["filename"]
            audio_path = f"/comfyui/output/{subfolder}/{filename}" if subfolder else f"/comfyui/output/{filename}"
            break

    if not audio_path:
        raise RuntimeError("TTS workflow did not produce audio output")

    # Upload to R2
    r2_key = f"{job_id}/audio/narration.mp3"
    audio_url = upload_to_r2(audio_path, r2_key)

    return {"audio_url": audio_url}


def action_align(job_input: dict) -> dict:
    """Run ForcedAligner + post-processing, return sections + captions."""
    audio_url = job_input["audio_url"]
    script = job_input["script"]
    fps = job_input.get("fps", 30)
    job_id = job_input.get("job_id", str(uuid.uuid4()))

    # Download audio
    audio_local = "/comfyui/input/narration.mp3"
    download_file(audio_url, audio_local)

    workflow = load_workflow("Qwen3-ASR.json")

    # Node 2 (LoadAudio): narration audio
    workflow["2"]["inputs"]["audio"] = "narration.mp3"

    prompt_id = queue_workflow(workflow)
    outputs = wait_for_completion(prompt_id, timeout=120)

    # Node 4 (PreviewAny) contains the alignment text
    # Node 1 output index 1 is the forced alignment text
    alignment_text = None
    node_4_output = outputs.get("4", {})
    if "text" in node_4_output:
        alignment_text = node_4_output["text"][0] if isinstance(node_4_output["text"], list) else node_4_output["text"]
    else:
        # Fallback: search all outputs
        for node_id, node_output in outputs.items():
            if "text" in node_output:
                text = node_output["text"]
                alignment_text = text[0] if isinstance(text, list) else text
                break

    if not alignment_text:
        raise RuntimeError("ForcedAligner workflow did not produce alignment output")

    # Post-processing
    from process_alignment import process
    result = process(
        aligner_text=alignment_text,
        script_text=script,
        fps=fps,
    )

    return result


# ── RunPod handler ──

def handler(job):
    job_input = job["input"]
    action = job_input.get("action")

    if action == "tts":
        return action_tts(job_input)
    elif action == "align":
        return action_align(job_input)
    else:
        raise ValueError(f"Unknown action: {action}. Expected 'tts' or 'align'.")


runpod.serverless.start({"handler": handler})
