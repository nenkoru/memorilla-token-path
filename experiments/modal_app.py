"""Run the experiment scripts on a Modal GPU instead of the Mac.

The scripts stay as they are (written for the Mac GPU): modal_entry.py maps the few Mac-only torch calls to CUDA and
pv4_port.DEV prefers CUDA when there is one. Code (experiments/, mport/, the patched memorilla repo) is mounted from
this folder at start, so an edit here is live on the next run. Data lives on the Modal volume "memorilla-port":

    /vol/work/ckpt, /vol/work/runs, /vol/work/ext/memorilla-demos   uploaded by `experiments/modal.sh push`
    /vol/hf                                                          Qwen3-8B and Qwen3-Embedding-4B (fetch_models)

Use experiments/modal.sh (push / models / run / pull); see its header. GPU type: env MPORT_GPU (default L40S, 48 GB).
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("MPORT_GPU", "L40S")
MODELS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-Embedding-4B"]
ENV = {"MPORT_SCRATCH": "/vol/work", "MEMORILLA_REPO": "/root/proj/repo/memorilla", "HF_HOME": "/vol/hf",
       "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "HF_XET_HIGH_PERFORMANCE": "1"}

app = modal.App("memorilla-port")
vol = modal.Volume.from_name("memorilla-port", create_if_missing=True)
skip = ["**/__pycache__", "**/*.pyc", "**/.DS_Store"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.15.0", "transformers==5.23.0", "sentence-transformers==6.3.0", "numpy==2.5.5",
                    "pyarrow==26.0.1", "safetensors==0.8.0", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
                    "accelerate==1.16.0", "scipy==1.18.1", "scikit-learn==1.9.1", "pandas==3.1.1", "datasets==5.2.0",
                    "tqdm", "hf-xet", "nltk==3.10.4", "rouge_score==0.1.2", "absl-py==2.5.1", "requests==2.34.2", "PyYAML==6.0.3")
    .env(ENV)
    .add_local_dir(ROOT / "experiments", "/root/proj/experiments", ignore=skip)
    .add_local_dir(ROOT / "mport", "/root/proj/mport", ignore=skip)
    .add_local_dir(ROOT / "work/repo/memorilla", "/root/proj/repo/memorilla", ignore=skip + ["**/.git"])
)


@app.function(image=image, volumes={"/vol": vol}, timeout=3600)
def fetch_models():
    """Download the two public Qwen models into the volume (once)."""
    from huggingface_hub import snapshot_download
    for m in MODELS:
        p = snapshot_download(m)
        print("fetched", m, p, flush=True)
    vol.commit()


@app.function(image=image, gpu=GPU, cpu=4, memory=49152, volumes={"/vol": vol}, timeout=3 * 3600, retries=0)
def run(script: str, args: str = "", name: str = ""):
    """Run experiments/<script> <args> on the GPU; output goes to stdout and /vol/work/logs/modal_<name>.log."""
    import shlex
    import subprocess
    import time
    name = name or Path(script).stem
    os.makedirs("/vol/work/logs", exist_ok=True)
    # Modal starts the call again from the top if its container is killed (out of memory, preemption). Never re-run:
    # a failing step would loop and keep a GPU busy. One attempt per call, marked on the volume.
    import modal as _modal
    mark = f"/vol/work/logs/modal_{name}.{_modal.current_function_call_id()}.started"
    if os.path.exists(mark):
        with open(f"/vol/work/logs/modal_{name}.log", "a") as log:
            log.write(f"MODAL_DONE {name} exit 137 gpu {GPU} (container was killed and restarted; not re-run)\n")
        vol.commit()
        return 137
    open(mark, "w").close()
    vol.commit()
    t0, code = time.time(), 0
    saved = t0
    with open(f"/vol/work/logs/modal_{name}.log", "w") as log:
        for step in args.split(" && ") if args else [""]:  # several steps of one script share the container
            cmd = ["python", "-I", "/root/proj/experiments/modal_entry.py", script, *shlex.split(step)]
            head = f"== {script} {step}  (+{(time.time() - t0) / 60:.1f} min)"
            print(head, flush=True)
            log.write(head + "\n")
            p = subprocess.Popen(cmd, cwd="/root/proj", env={**os.environ, "HF_HUB_OFFLINE": "1"}, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                if "Loading weights" in line:
                    continue
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if time.time() - saved > 60:  # keep the volume copy of the log and partial results fresh
                    vol.commit()
                    saved = time.time()
            code = p.wait()
            vol.commit()
            if code:
                break
        tail = f"MODAL_DONE {name} exit {code} gpu {GPU} minutes {(time.time() - t0) / 60:.1f}"
        print(tail, flush=True)
        log.write(tail + "\n")
    vol.commit()
    return code


@app.local_entrypoint()
def main(script: str, args: str = "", name: str = ""):
    # spawn and return: with `modal run -d` a spawned call keeps running after this client exits. (A blocking
    # run.remote() here stopped the app when the Mac lost the network, 2026-11-22 23:07.)
    call = run.spawn(script, args, name)
    print("spawned", call.object_id, flush=True)
