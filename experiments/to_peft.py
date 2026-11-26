"""Convert the scripts' LoRA files ({"<i>.A", "<i>.B"}, i = layer * 4 + q/k/v/o) to a PEFT adapter folder for Qwen3-8B.

    to_peft.py lora_kd27a.pt out_dir [config_json]
The reverse is in the model card on Hugging Face.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = sd.get("lora", sd) if isinstance(sd, dict) else sd
    n = len([k for k in sd if k.endswith(".A")])
    assert n == 36 * 4, f"expected 144 LoRA pairs, got {n}: {list(sd)[:4]}"
    st = {}
    for i in range(n):
        L, p = divmod(i, 4)
        base = f"base_model.model.model.layers.{L}.self_attn.{NAMES[p]}"
        st[f"{base}.lora_A.weight"] = sd[f"{i}.A"].float().contiguous()
        st[f"{base}.lora_B.weight"] = sd[f"{i}.B"].float().contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(st, out / "adapter_model.safetensors")
    if len(sys.argv) > 3:
        shutil.copy(sys.argv[3], out / "adapter_config.json")
    r = st[next(iter(st))].shape[0]
    print(json.dumps({"src": src.name, "pairs": n, "rank": r, "out": str(out)}))


if __name__ == "__main__":
    main()
