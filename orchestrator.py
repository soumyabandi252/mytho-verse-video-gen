"""
orchestrator.py
Runs on GitHub Actions (CPU only, no GPU needed). Talks to the Kaggle API to:
  1. Push a kernel (kaggle_kernel_generate.py) with the user's prompt baked in.
  2. Poll until the Kaggle GPU kernel finishes running.
  3. Download the kernel's output.mp4 back into the repo/workflow artifact.

Requires:
  pip install kaggle
  Kaggle API token as env vars: KAGGLE_USERNAME, KAGGLE_KEY
  (Store these as GitHub Actions secrets, never commit them.)

Usage:
  python orchestrator.py --prompt "Krishna and a baby playing near the river at sunset" \
                          --kaggle-user your_kaggle_username \
                          --slug mythoverse-video-gen
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

KERNEL_SCRIPT = "kaggle_kernel_generate.py"
POLL_INTERVAL_SECONDS = 60
MAX_WAIT_SECONDS = 3 * 60 * 60  # 3 hours safety cap


def build_kernel_folder(work_dir: Path, kaggle_user: str, slug: str, prompt: str):
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(KERNEL_SCRIPT, work_dir / KERNEL_SCRIPT)

    metadata = {
        "id": f"{kaggle_user}/{slug}",
        "title": slug,
        "code_file": KERNEL_SCRIPT,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(work_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Embed the prompt directly into the script since Kaggle kernels don't
    # accept custom runtime env vars via the push API.
    script_path = work_dir / KERNEL_SCRIPT
    content = script_path.read_text()
    content = content.replace(
        'PROMPT = os.environ.get(\n    "SCENE_PROMPT",\n'
        '    "Krishna playing a flute in a moonlit forest while a baby crawls toward him laughing"\n)',
        f'PROMPT = {json.dumps(prompt)}'
    )
    script_path.write_text(content)


def push_kernel(work_dir: Path):
    subprocess.run(["kaggle", "kernels", "push", "-p", str(work_dir)], check=True)


def wait_for_completion(kaggle_user: str, slug: str):
    kernel_ref = f"{kaggle_user}/{slug}"
    elapsed = 0
    while elapsed < MAX_WAIT_SECONDS:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture_output=True, text=True
        )
        status_line = result.stdout.strip()
        print(f"[{elapsed}s] {status_line}")
        if '"complete"' in status_line or "has status \"complete\"" in status_line:
            return True
        if "error" in status_line.lower():
            raise RuntimeError(f"Kaggle kernel failed: {status_line}")
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
    raise TimeoutError("Kaggle kernel did not finish within the max wait window.")


def download_output(kaggle_user: str, slug: str, dest_dir: Path):
    kernel_ref = f"{kaggle_user}/{slug}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_ref, "-p", str(dest_dir)],
        check=True
    )
    mp4_path = dest_dir / "output.mp4"
    if not mp4_path.exists():
        raise FileNotFoundError("output.mp4 not found in kernel output.")
    return mp4_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--kaggle-user", required=True)
    parser.add_argument("--slug", default="mythoverse-video-gen")
    args = parser.parse_args()

    work_dir = Path("kaggle_job")
    build_kernel_folder(work_dir, args.kaggle_user, args.slug, args.prompt)
    push_kernel(work_dir)
    wait_for_completion(args.kaggle_user, args.slug)
    output_path = download_output(args.kaggle_user, args.slug, Path("output"))
    print(f"Video ready at: {output_path}")


if __name__ == "__main__":
    main()
