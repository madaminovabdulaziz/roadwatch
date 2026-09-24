# Running on Kaggle (T4)

Kaggle gives us a free Linux machine with the same GPU as the organizers (T4, 16 GB) but only 4 CPU
cores, against their 8. Decode timings here are therefore pessimistic.

## One-time setup

1. Kaggle account with a **verified phone number** (Settings → Phone verification). Without it there
   is no GPU and no internet in notebooks.
2. GitHub fine-grained token for the private repo (https://github.com/settings/personal-access-tokens/new):
   Repository access → *Only select repositories* → `roadwatch`. Then under Permissions click
   **Add permissions → Contents** and set it to *Read-only*; without it the clone fails with a misleading
   "Write access to repository not granted" 403. Expiry 7 days.
3. In the notebook: Add-ons → Secrets → add `GITHUB_TOKEN` with that token and attach it to the notebook.
4. Notebook settings: Accelerator **GPU T4**, Internet **On**.

## Cell 1: clone (the token never appears in output, even on failure)

```python
from kaggle_secrets import UserSecretsClient
import subprocess

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo = "github.com/madaminovabdulaziz/roadwatch.git"
subprocess.run(["rm", "-rf", "/tmp/roadwatch"], check=True)
r = subprocess.run(["git", "clone", "-q", f"https://x-access-token:{token}@{repo}", "/tmp/roadwatch"],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError("git clone failed:\n" + r.stderr.replace(token, "***"))
subprocess.run(["git", "-C", "/tmp/roadwatch", "remote", "set-url", "origin", f"https://{repo}"], check=True)
print(subprocess.run(["git", "-C", "/tmp/roadwatch", "log", "--oneline", "-3"],
                     capture_output=True, text=True).stdout)
```

## Cell 2: clean-machine check

```python
!bash /tmp/roadwatch/kaggle/check_env.sh
```

It installs `requirements.txt` into a fresh venv with plain pip, checks CUDA on the T4, runs the
tests, and times the official harness on a synthetic clip in the sample format. It takes about 5–10 minutes.

## Cell 3: download the samples and benchmark decoding (RUNBOOK P0.2)

Re-run Cell 1 first to get the latest code. The samples come from our own Drive copies listed in
`kaggle/samples.tsv`: the organizers' shared links hit Drive's download quota, so each teammate
made a copy (Drive → Shared with me → right-click → Make a copy) shared as "Anyone with the link".

```python
!bash /tmp/roadwatch/kaggle/setup_venv.sh && bash /tmp/roadwatch/kaggle/fetch_samples.sh
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python scripts/bench.py /tmp/samples --seconds 30 --verify --json /kaggle/working/bench_decode.json
```

The bench prints every decode mode as "x dur" (wall time / video time, the unit of the 3x budget)
and checks that the fast path's frames line up with the harness's frames.

## Cell 4: weights, tests on the GPU, track cache, perception benchmark (RUNBOOK P0.3)

Re-run Cell 1 first. `weights/download.sh` is the organizers' path; while the repository is private it
uses the `GITHUB_TOKEN` secret to read the release.

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
!bash /tmp/roadwatch/kaggle/setup_venv.sh && bash /tmp/roadwatch/kaggle/fetch_samples.sh && bash /tmp/roadwatch/weights/download.sh
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python -m pytest -p no:cacheprovider
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python scripts/bench.py /tmp/samples --seconds 30 --modes harness --workers "" --perception --check-fp16 --json /kaggle/working/bench_perception_960.json
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python scripts/bench.py /tmp/samples --seconds 30 --modes "" --workers "" --perception --imgsz 1280 --json /kaggle/working/bench_perception_1280.json
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python scripts/cache_tracks.py /tmp/samples --out /kaggle/working/tracks
```

Download `/kaggle/working/tracks/*.parquet` + `*.json` (Output panel) into `cache/tracks/` locally.

