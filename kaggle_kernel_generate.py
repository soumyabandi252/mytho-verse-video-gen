"""
kaggle_kernel_generate.py
Runs INSIDE a Kaggle Notebook (GPU: T4 x2 recommended - NOT P100, see README).
Generates a ~60s mythological video: Lord Krishna + a baby, from one text prompt.

Pipeline:
  1. Install/upgrade required packages (Kaggle's base image is often outdated/missing them).
  2. Generate two reference images (Krishna, baby) with FLUX.1-schnell (memory-optimized for T4 16GB).
  3. Split the user prompt into N scenes.
  4. For each scene, generate a 6-10s clip with CogVideoX-5B-I2V, conditioned
     on the reference images so characters stay visually consistent.
  5. Generate narration audio with a transformers-native TTS model (facebook/mms-tts-eng),
     with a pitch/speed shift applied afterward for a distinct voice.
  6. Stitch everything with ffmpeg into output.mp4 (Kaggle kernel output).

IMPORTANT Kaggle settings required (set manually in the Kaggle web UI, not via API):
  - Open this kernel in the Kaggle editor, open session settings, and set
    Accelerator = "GPU T4 x2". Do NOT use "GPU P100" - Kaggle's current
    default PyTorch build has dropped support for the P100's older
    CUDA compute capability (sm_60), causing GPU init to fail.
  - Internet: ON (requires phone verification on your Kaggle account).
  - Kernel type: "Script", so it runs headless via `kaggle kernels push`.

Notes on T4 (16GB VRAM) memory management:
  - Uses float16 (not bfloat16) - T4 is a Turing GPU without proper bf16 tensor core support.
  - Uses enable_model_cpu_offload() + enable_attention_slicing() + vae slicing/tiling on
    BOTH diffusion pipelines so no single pipeline tries to hold its full weights on GPU at once.

Note on TTS engine:
  - We intentionally do NOT use the separate "TTS"/"coqui-tts" packages. Their internal
    code depends on specific transformers internals that conflict with the newer
    transformers version diffusers/CogVideoX requires (import errors like
    "cannot import name isin_mps_friendly"). Using transformers' own TTS pipeline
    (facebook/mms-tts-eng) avoids that whole class of dependency conflicts since it
    shares the same transformers install as the video pipeline.
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
from diffusers import FluxPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import pipeline as hf_pipeline

PROMPT = os.environ.get(
    "SCENE_PROMPT",
    "Krishna playing a flute in a moonlit forest while a baby crawls toward him laughing"
)
NUM_SCENES = 6          # 6 scenes x ~10s = ~60s total
CLIP_SECONDS = 10
FPS = 8
WORKDIR = "/kaggle/working"
os.makedirs(WORKDIR, exist_ok=True)

DTYPE = torch.float16  # T4-safe dtype


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def split_into_scenes(prompt: str, n: int) -> list:
    base = prompt.strip().rstrip(".")
    beats = [
        f"Opening shot: {base}, wide establishing view",
        f"{base}, close-up on Krishna's face, gentle smile",
        f"{base}, close-up on the baby reacting joyfully",
        f"{base}, medium shot showing both characters together",
        f"{base}, dramatic lighting shift, divine glow intensifies",
        f"Closing shot: {base}, camera slowly pulls back, peaceful ending",
    ]
    return beats[:n] if n <= len(beats) else (beats * ((n // len(beats)) + 1))[:n]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. Enable GPU T4 x2 in Kaggle kernel session settings.")
        raise RuntimeError("No GPU available - this pipeline requires a GPU kernel.")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name}, VRAM: {total_vram:.1f} GB")
    if "P100" in gpu_name:
        raise RuntimeError(
            "This kernel is running on a P100, which is incompatible with the current "
            "PyTorch build (dropped sm_60 support). Open this kernel in the Kaggle "
            "editor and switch Accelerator to 'GPU T4 x2' in session settings, then re-run."
        )

    print("Loading FLUX.1-schnell for reference images (memory-optimized)...")
    flux = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=DTYPE
    )
    flux.enable_model_cpu_offload()
    flux.enable_attention_slicing()
    flux.vae.enable_slicing()
    flux.vae.enable_tiling()

    krishna_ref_path = f"{WORKDIR}/krishna_ref.png"
    baby_ref_path = f"{WORKDIR}/baby_ref.png"

    krishna_img = flux(
        "Lord Krishna, blue-skinned deity, peacock feather crown, yellow silk dhoti, "
        "playing a flute, serene expression, soft divine lighting, front-facing portrait, "
        "highly detailed digital painting style",
        num_inference_steps=4, guidance_scale=0.0,
        height=512, width=512,
    ).images[0]
    krishna_img.save(krishna_ref_path)

    baby_img = flux(
        "A happy chubby baby, warm golden lighting, sitting pose, front-facing portrait, "
        "soft skin texture, wearing simple traditional Indian baby clothes, "
        "highly detailed digital painting style",
        num_inference_steps=4, guidance_scale=0.0,
        height=512, width=512,
    ).images[0]
    baby_img.save(baby_ref_path)

    del flux
    free_gpu()

    scenes = split_into_scenes(PROMPT, NUM_SCENES)
    print("Scenes:", json.dumps(scenes, indent=2))

    print("Loading CogVideoX-5B-I2V (memory-optimized)...")
    video_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", torch_dtype=DTYPE
    )
    video_pipe.enable_model_cpu_offload()
    video_pipe.enable_attention_slicing()
    video_pipe.vae.enable_slicing()
    video_pipe.vae.enable_tiling()

    clip_paths = []
    for i, scene_prompt in enumerate(scenes):
        ref_image = load_image(krishna_ref_path if i % 2 == 0 else baby_ref_path)
        print(f"Generating scene {i+1}/{len(scenes)}: {scene_prompt}")
        frames = video_pipe(
            prompt=scene_prompt,
            image=ref_image,
            num_videos_per_prompt=1,
            num_inference_steps=25,
            num_frames=CLIP_SECONDS * FPS,
            guidance_scale=6.0,
        ).frames[0]
        clip_path = f"{WORKDIR}/scene_{i:02d}.mp4"
        export_to_video(frames, clip_path, fps=FPS)
        clip_paths.append(clip_path)
        free_gpu()

    del video_pipe
    free_gpu()

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

    # Pitch/speed shift for a distinct narration voice
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
