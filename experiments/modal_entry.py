"""Entry point inside the Modal container: map the Mac-only torch calls to CUDA, then run one experiment script as
__main__ with its arguments. Usage: python -I modal_entry.py <script.py> <args...>"""

import runpy
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.mps.empty_cache = torch.cuda.empty_cache
    torch.mps.synchronize = torch.cuda.synchronize
    torch.mps.current_allocated_memory = torch.cuda.memory_allocated
    torch.mps.driver_allocated_memory = torch.cuda.memory_reserved
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"device {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB", flush=True)
else:
    print("no CUDA device: running on CPU", flush=True)

here = Path(__file__).resolve().parent
script = here / sys.argv[1]
sys.argv = [str(script)] + sys.argv[2:]
sys.path.insert(0, str(here))
runpy.run_path(str(script), run_name="__main__")
