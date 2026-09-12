# Mytho Verse Video Gen

Free-tier, GPU-less pipeline that generates ~60 second mythological videos (Lord Krishna + a baby) from any single text prompt. No paid APIs, no local GPU required — all heavy compute runs on Kaggle's free GPU quota, orchestrated by GitHub Actions.

## How it works

1. You trigger the `Generate Mythology Video` GitHub Actions workflow with a text prompt.
2. `orchestrator.py` (runs on the GitHub Actions CPU runner) pushes `kaggle_kernel_generate.py` to a Kaggle GPU kernel via the Kaggle API, with your prompt embedded.
3. On Kaggle's free GPU (T4/P100):
   - FLUX.1-schnell generates one reference image each for Krishna and the baby, used to keep their appearance consistent across scenes.
   - The prompt is split into 6 scenes (~10s each, ~60s total).
   - CogVideoX-5B-I2V generates each scene, conditioned on the reference images.
   - Coqui XTTS-v2 generates narration audio, with a slight speed/pitch shift applied for a distinct voice.
   - ffmpeg stitches all clips and narration into `output.mp4`.
4. The orchestrator polls Kaggle until the kernel finishes, downloads `output.mp4`, and GitHub Actions uploads it as a workflow artifact for you to download and post manually.

## Setup

1. Create a free [Kaggle](https://www.kaggle.com/) account.
2. Go to Kaggle Account Settings → API → "Create New Token" to get your `username` and `key`.
3. In this repo's GitHub Settings → Secrets and variables → Actions, add:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
4. Go to the Actions tab, select "Generate Mythology Video", click "Run workflow", and enter your prompt.
5. Wait for the run to finish (can take 30–90+ minutes depending on Kaggle GPU queue and model download time on first run), then download `output.mp4` from the workflow's Artifacts section.

## Known limitations

- Kaggle's free tier gives ~30 GPU-hours/week — not literally infinite generations, but no per-video cost.
- Character consistency (Krishna/baby) is approximate; open-weight models don't have dedicated face-lock features like paid tools (Kling, Veo).
- Voice quality from XTTS-v2 is good but behind paid options like ElevenLabs. Pass a `speaker_wav` in `kaggle_kernel_generate.py` to clone a specific voice.
- First run is slow due to ~15GB of model weight downloads (cached afterward on the same Kaggle kernel).
- Uploading the finished video to Instagram/YouTube is manual for now.
