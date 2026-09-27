# RoadWatch: final Kaggle run

About **1.5 hours**, mostly waiting. It produces the two files we submit and publish:

- `predictions_samples_unpaced.json`: our output on the 3 sample videos (the deliverable)
- `kaggle_t4_official.json`: the organizers' official run on the same machine (timing proof)

## Before you start (one time)

1. **Verified phone number** on your Kaggle account (Settings → Phone verification). Without it there is no
   GPU and no internet in notebooks.
2. **GitHub token** that can read `madaminovabdulaziz/roadwatch` (you are a collaborator). Use a
   **classic** token: github.com → Settings → Developer settings → Personal access tokens →
   **Tokens (classic)** → Generate new token → tick the **`repo`** scope → Generate, then copy it.
   Fine-grained tokens cannot reach another person's private repository.
3. **New Kaggle notebook.** In the right panel:
   - **Accelerator:** GPU T4
   - **Internet:** On
   - **Add-ons → Secrets → Add secret:** name `GITHUB_TOKEN`, value = your token. Make sure it is
     **attached** to this notebook (the checkbox next to it).

## Run these 3 cells in order

### Cell 1: get the code

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

✅ The first printed line starts with **`d6f00ec`** or a newer commit.

### Cell 2: install and download the 3 videos (about 10 min)

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
if not os.path.exists("/tmp/rw-venv/bin/python"):
    !bash /tmp/roadwatch/kaggle/setup_venv.sh
!bash /tmp/roadwatch/kaggle/fetch_samples.sh && bash /tmp/roadwatch/weights/download.sh
```

✅ It lists **C3896.MP4, C3902.MP4, C3905.MP4** (5.9 G, 5.5 G, 2.2 G) and ends with
`yolo11m.json: present`. An `ensurepip` error or an `Error in sitecustomize ... wrapt` warning near the
top is harmless.

### Cell 3: the two final runs, with download links (about 75 min)

```python
import os, json
from IPython.display import FileLink, display
os.chdir("/kaggle/working")
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python scripts/make_predictions_samples.py /tmp/samples --unpaced --out /kaggle/working/predictions_samples_unpaced.json
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python evaluate.py --pred /kaggle/working/predictions_samples_unpaced.json --validate-only
display(FileLink("predictions_samples_unpaced.json"))
!cd /tmp/roadwatch && env -u PYTHONPATH /tmp/rw-venv/bin/python run_submission.py --videos /tmp/samples --out /kaggle/working/kaggle_t4_official.json --team roadwatch
for name in ["predictions_samples_unpaced.json", "kaggle_t4_official.json"]:
    for video, log in json.load(open(name))["log"].items():
        print(name, video, f"{log['total_sec'] / log['duration']:.2f}x", log["errors"])
display(FileLink("kaggle_t4_official.json"))
```

## ⚠️ Important, or it has to be redone

- **Keep the tab open and check back every 10–15 minutes.** Kaggle wipes everything (code, videos and
  results) if the notebook sits idle after a cell finishes.
- After **about 40 min**, a blue link **`predictions_samples_unpaced.json`** appears under Cell 3.
  **Download it right away**: click it, or right-click → Save Link As…
- After **about 75 min**, a second link **`kaggle_t4_official.json`** appears. Download it too.

## ✅ What a good result looks like

- `format: 3 video(s), ... 0 error(s) ... -> VALID`
- Each of the final printed lines ends with `[]` (no errors).
- The three `kaggle_t4_official.json` lines are **under 3.00x** (about 2.7x is expected).

## Send back

1. `predictions_samples_unpaced.json`
2. `kaggle_t4_official.json`
3. A copy or screenshot of Cell 3's last printed lines

If anything fails, send a screenshot of the error; don't change the code.
