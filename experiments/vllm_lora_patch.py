"""Start production's vLLM build (a V100 vLLM build fork, used read-only) with a LoRA fix applied in memory only.

The fork added LogitsProcessor._apply_logits_transforms (and friends) but LogitsProcessorWithLoRA, which wraps it when
--enable-lora is set, does not have them, so the engine dies at start-up. Here the missing methods are copied onto the
wrapper class before the CLI starts; the engine process is forked from this one, so it sees the same fix.
Nothing on disk in the production environment changes.

  ~/<vllm-venv>/bin/python experiments/vllm_lora_patch.py serve <model> ... (same arguments as `vllm serve`)
"""
import os
import sys

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

from vllm.lora.layers.logits_processor import LogitsProcessorWithLoRA  # noqa: E402
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402

added = []
for name, fn in vars(LogitsProcessor).items():
    if callable(fn) and not name.startswith("__") and not hasattr(LogitsProcessorWithLoRA, name):
        setattr(LogitsProcessorWithLoRA, name, fn)
        added.append(name)
print("vllm_lora_patch: added to LogitsProcessorWithLoRA:", ", ".join(added) or "nothing", flush=True)

from vllm.entrypoints.cli.main import main  # noqa: E402

sys.argv = ["vllm"] + sys.argv[1:]
sys.exit(main())
