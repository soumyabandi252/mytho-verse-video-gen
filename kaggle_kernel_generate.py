"""
kaggle_kernel_generate.py
Runs INSIDE a Kaggle Notebook (GPU: T4 x2, forced via orchestrator.py's machine_shape).
Generates a ~60s mythological video: Lord Krishna + a baby, from one text prompt.

Pipeline:
  1. Install/upgrade required packages (Kaggle's base image is often outdated/missing them).
  2. Generate two reference images (Krishna, baby) with SDXL-Turbo (open, ungated, T4-light).
  3. Split the user prompt into N scenes.
  4. For each scene, animate a reference image into a ~4s clip with Stable Video Diffusion
     (img2vid-xt), alternating between the Krishna and baby reference images so both
     characters stay visually consistent across the whole video.
  5. Generate narration audio with a transformers-native TTS model (facebook/mms-tts-eng),
     with a pitch/speed shift applied afterward for a distinct voice.
  6. Stitch everything with ffmpeg into output.mp4 (Kaggle kernel output).

Kaggle settings required:
  - GPU type is forced to T4 via "machine_shape" in orchestrator.py's kernel-metadata.json.
  - Internet: ON (requires phone verification on your Kaggle account).
  - Kernel type: "Script", so it runs headless via `kaggle kernels push`.

IMPORTANT - why CogVideoX-5B was replaced with Stable Video Diffusion:
  CogVideoX-5B-I2V ships a 4.7B-parameter T5-XXL text encoder plus a 5B video transformer.
  Even with fp16 weights and low_cpu_mem_usage, loading its checkpoint shards into host
  RAM before GPU offload consistently exceeded Kaggle's free-tier ~13GB RAM limit and got
  the process OS-killed (twice, confirmed via logs, including after the fp16 variant
  request silently fell back to full-size weights since no fp16 variant is published for
  that repo). Stable Video Diffusion (img2vid-xt) is roughly 8x smaller, is natively an
  image-to-video model (a natural fit for animating our reference images), and is one of
  the most widely used models specifically proven to run on free Kaggle T4 kernels.

Notes on T4 (16GB VRAM) memory management:
  - Uses float16 (not bfloat16) - T4 is a Turing GPU without proper bf16 tensor core support.
  - Uses enable_model_cpu_offload() + vae slicing/tiling so pipelines don't hold their full
    weights on GPU at once.

Note on image model:
  - FLUX.1-schnell became a gated Hugging Face model (requires login + accepted license),
    which would need an extra manual HF token setup step. We use "stabilityai/sdxl-turbo"
    instead - fully open, no login required, and lighter on VRAM.

Note on TTS engine:
  - We intentionally do NOT use the separate "TTS"/"coqui-tts" packages. Their internal
    code depends on specific transformers internals that conflict with the newer
    transformers version diffusers requires (import errors like
    "cannot import name isin_mps_friendly"). Using transformers' own TTS pipeline
    (facebook/mms-tts-eng) avoids that whole class of dependency conflicts.
"""

import subprocess
import sys

print("Installing/upgrading required packages...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "diffusers>=0.31.0", "transformers>=4.44.0", "accelerate>=0.33.0",
     "imageio-ffmpeg", "sentencepiece", "protobuf", "scipy"],
    check=True
)

import os
import json
import traceback
import gc

import torch
import scipy.io.wavfile
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
from transformers import pipeline as hf_pipeline

PROMPT = os.environ.get(
    "SCENE_PROMPT",
    "Krishna playing a flute in a moonlit forest while a baby crawls toward him laughing"
)
NUM_SCENES = 15         # 15 clips x ~4s (25 frames @ 6fps) = ~60s total
SVD_NUM_FRAMES = 25
SVD_FPS = 6
WORKDIR = "/kaggle/working"
os.makedirs(WORKDIR, exist_ok=True)

DTYPE = torch.float16  # T4-safe dtype


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected.")
        raise RuntimeError("No GPU available - this pipeline requires a GPU kernel.")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name}, VRAM: {total_vram:.1f} GB")
    if "P100" in gpu_name:
        raise RuntimeError(
            "This kernel is running on a P100, which is incompatible with the current "
            "PyTorch build (dropped sm_60 support). Check machine_shape in orchestrator.py."
        )

    print("Loading SDXL-Turbo for reference images (open, ungated, memory-optimized)...")
    img_pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=DTYPE, variant="fp16", low_cpu_mem_usage=True
    )
    img_pipe.enable_model_cpu_offload()
    img_pipe.vae.enable_slicing()
    img_pipe.vae.enable_tiling()

    krishna_ref_path = f"{WORKDIR}/krishna_ref.png"
    baby_ref_path = f"{WORKDIR}/baby_ref.png"

    krishna_img = img_pipe(
        "Lord Krishna, blue-skinned deity, peacock feather crown, yellow silk dhoti, "
        "playing a flute, serene expression, soft divine lighting, front-facing portrait, "
        "highly detailed digital painting style",
        num_inference_steps=2, guidance_scale=0.0,
        height=576, width=1024,
    ).images[0]
    krishna_img.save(krishna_ref_path)

    baby_img = img_pipe(
        "A happy chubby baby, warm golden lighting, sitting pose, front-facing portrait, "
        "soft skin texture, wearing simple traditional Indian baby clothes, "
        "highly detailed digital painting style",
        num_inference_steps=2, guidance_scale=0.0,
        height=576, width=1024,
    ).images[0]
    baby_img.save(baby_ref_path)

    del img_pipe
    free_gpu()
    gc.collect()

    print("Loading Stable Video Diffusion (img2vid-xt, T4-light)...")
    video_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=DTYPE, variant="fp16", low_cpu_mem_usage=True
    )
    video_pipe.enable_model_cpu_offload()

    clip_paths = []
    for i in range(NUM_SCENES):
        ref_path = krishna_ref_path if i % 2 == 0 else baby_ref_path
        ref_image = load_image(ref_path).resize((1024, 576))
        print(f"Generating scene {i+1}/{NUM_SCENES} from {os.path.basename(ref_path)}")
        frames = video_pipe(
            ref_image,
            num_frames=SVD_NUM_FRAMES,
            decode_chunk_size=8,
            motion_bucket_id=110,
            noise_aug_strength=0.02,
        ).frames[0]
        clip_path = f"{WORKDIR}/scene_{i:02d}.mp4"
        export_to_video(frames, clip_path, fps=SVD_FPS)
        clip_paths.append(clip_path)
        free_gpu()

    del video_pipe
    free_gpu()
    gc.collect()

    print("Loading transformers TTS pipeline (facebook/mms-tts-eng)...")
    tts_pipe = hf_pipeline("text-to-speech", model="facebook/mms-tts-eng", device=0)

    narration_text = (
        f"{PROMPT}. In the gentle glow of dusk, the divine and the innocent came together, "
        "a moment of pure devotion and joy."
    )
    narration_path = f"{WORKDIR}/narration.wav"
    tts_output = tts_pipe(narration_text)
    scipy.io.wavfile.write(
        narration_path,
        rate=tts_output["sampling_rate"],
        data=tts_output["audio"][0]
    )

    del tts_pipe
    free_gpu()

    shifted_narration = f"{WORKDIR}/narration_shifted.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", narration_path,
        "-af", "asetrate=44100*0.95,aresample=44100,atempo=1.05",
        shifted_narration
    ], check=True)

    concat_list_path = f"{WORKDIR}/concat_list.txt"
    with open(concat_list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    silent_full_path = f"{WORKDIR}/full_silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path, "-c", "copy", silent_full_path
    ], check=True)

    final_output = f"{WORKDIR}/output.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", silent_full_path, "-i", shifted_narration,
        "-c:v", "copy", "-c:a", "aac", "-shortest", final_output
    ], check=True)

    print(f"Done. Final video at: {final_output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL ERROR IN PIPELINE:")
        traceback.print_exc()
        sys.exit(1)
