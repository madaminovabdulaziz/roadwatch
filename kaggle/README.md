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
