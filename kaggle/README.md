# Running on Kaggle (T4)

Kaggle gives us a free Linux machine with the same GPU as the organizers (T4, 16 GB) but only 4 CPU
cores, against their 8. Decode timings here are therefore pessimistic.

## One-time setup

1. Kaggle account with a **verified phone number** (Settings → Phone verification). Without it there
   is no GPU and no internet in notebooks.
2. GitHub fine-grained token for the private repo: GitHub → Settings → Developer settings →
   Fine-grained tokens → Generate. Repository access: *Only select repositories* → `roadwatch`;
   Permissions → Contents: *Read-only*; expiry 7 days.
3. In the notebook: Add-ons → Secrets → add `GITHUB_TOKEN` with that token and attach it to the notebook.
4. Notebook settings: Accelerator **GPU T4**, Internet **On**.

## Cell 1: clone (token stays out of saved output)

```python
from kaggle_secrets import UserSecretsClient
import subprocess

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(["rm", "-rf", "/tmp/roadwatch"], check=True)
subprocess.run(["git", "clone", "-q", f"https://{token}@github.com/madaminovabdulaziz/roadwatch.git",
                "/tmp/roadwatch"], check=True)
subprocess.run(["git", "-C", "/tmp/roadwatch", "remote", "set-url", "origin",
                "https://github.com/madaminovabdulaziz/roadwatch.git"], check=True)
print(subprocess.run(["git", "-C", "/tmp/roadwatch", "log", "--oneline", "-3"],
                     capture_output=True, text=True).stdout)
```

## Cell 2: clean-machine check

```python
!bash /tmp/roadwatch/kaggle/check_env.sh
```

It installs `requirements.txt` into a fresh venv with plain pip, checks CUDA on the T4, runs the
tests, and times the official harness on a synthetic clip in the sample format. It takes about 5–10 minutes.
