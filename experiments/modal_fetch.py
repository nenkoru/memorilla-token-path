"""CPU-only Modal app: download the two public Qwen models into the volume (once). Kept apart from modal_app.py so it
runs without a GPU function in the app. Usage: work/venv-modal/bin/modal run experiments/modal_fetch.py"""

import modal

app = modal.App("memorilla-port-fetch")
vol = modal.Volume.from_name("memorilla-port", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub==2.4.0", "hf-xet").env(
    {"HF_HOME": "/vol/hf", "HF_XET_HIGH_PERFORMANCE": "1"})


@app.function(image=image, volumes={"/vol": vol}, timeout=3600)
def fetch():
    from huggingface_hub import snapshot_download
    for m in ["Qwen/Qwen3-8B", "Qwen/Qwen3-Embedding-4B"]:
        print("fetched", m, snapshot_download(m), flush=True)
    vol.commit()


@app.local_entrypoint()
def main():
    fetch.remote()
