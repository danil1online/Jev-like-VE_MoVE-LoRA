"""
Common utilities: dtype handling, device detection, directories, logging, downloads.
"""
import os
import sys
import json
import time
import fcntl
import logging
import requests
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Global dtype handling
# COMPUTE_DTYPE: the dtype used for model weights and computation.
# By default, bf16 if available, else fp32 (e.g. on CPU).
# Can be overridden with the JEVELIKE_DTYPE environment variable:
#   JEVELIKE_DTYPE=fp32 python train.py
#   JEVELIKE_DTYPE=bf16 python train.py
#   JEVELIKE_DTYPE=fp16 python train.py
# -----------------------------------------------------------------------------

def parse_dtype(s):
    s = s.lower().strip()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {s!r} (use 'fp32', 'bf16', or 'fp16')")

_dtype_override = os.environ.get("JEVELIKE_DTYPE")
if _dtype_override is not None:
    COMPUTE_DTYPE = parse_dtype(_dtype_override)
    print(f"Using dtype override: {COMPUTE_DTYPE}", file=sys.stderr)
else:
    COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def compute_init(device_type, tensor_cache_size_gib=0):
    """Initialize the compute environment for the given device type."""
    if device_type == "cuda":
        if tensor_cache_size_gib > 0:
            free, total = torch.cuda.mem_get_info()
            desired_free = tensor_cache_size_gib * 1024**3
            if free < desired_free:
                print(f"CUDA cache evicted: {free / 1024**3:.2f} GiB free -> evicting to reach {desired_free / 1024**3:.2f} GiB free")
                torch.cuda.empty_cache()
                free = torch.cuda.mem_get_info()[0]
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    elif device_type == "cpu":
        torch.set_num_threads(1)
        torch.manual_seed(42)
    else:
        raise ValueError(f"Unknown device type: {device_type}")

def compute_cleanup(device_type):
    """Clean up the compute environment (e.g. empty CUDA cache)."""
    if device_type == "cuda":
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------

def get_base_dir():
    """The base directory where checkpoints and tokenizers are stored,
    as an absolute path. By default the current working directory."""
    base_dir = os.path.abspath(os.curdir)
    return base_dir

def get_data_dir():
    """The directory where the training data is stored, as an absolute path.
    Defaults to <base_dir>/data (overridable with JEVELIKE_DATA_DIR)."""
    data_dir = os.environ.get("JEVELIKE_DATA_DIR")
    if data_dir is None:
        data_dir = os.path.join(get_base_dir(), "data")
    return os.path.abspath(data_dir)

def get_checkpoints_dir():
    return os.path.join(get_base_dir(), "checkpoints")

def get_runs_dir():
    return os.path.join(get_base_dir(), "runs")

# -----------------------------------------------------------------------------
# Distributed utilities (single-process by default; no-op if not in a group)
# -----------------------------------------------------------------------------

def is_dist_priority_active():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_dist_info():
    """(ddp_active, ddp_rank, ddp_local_rank, ddp_world_size). Single-process by default."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        return True, dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    return False, 0, 0, 1

def print0(*messages, sep=" ", file=None):
    """Prints a message to stdout from one process only (rank 0, or the only process)."""
    if isinstance(messages, tuple):
        messages = list(messages)
    if not isinstance(messages, list):
        raise ValueError("messages must be a list or a tuple")
    for i, message in enumerate(messages):
        if i > 0:
            print(sep, end="", file=file)
        print(message, end="", file=file)
    if is_dist_priority_active() and dist.get_rank() == 0:
        print(file=file)
    elif not is_dist_priority_active():
        print(file=file)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_default_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s-%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        stream=sys.stderr,
        force=True,
    )

def get_logger(name):
    return logging.getLogger(name)

class JsonlLogger:
    """Append-only JSONL run logger (replaces wandb for this project)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def log(self, **kwargs):
        entry = {"step": int(time.time() - self.t0), **kwargs}
        self.file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

def download_file(url, dest, chunk_size=1024 * 1024):
    """Stream a file from `url` to `dest` with a file lock and resume support."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    lock_path = dest + ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            headers = {}
            resume_pos = 0
            if os.path.exists(dest):
                resume_pos = os.path.getsize(dest)
                if resume_pos > 0:
                    headers["Range"] = f"bytes={resume_pos}-"
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            if response.status_code == 416:
                print(f"File already complete: {dest}")
                return
            response.raise_for_status()
            mode = "ab" if resume_pos > 0 and response.status_code == 206 else "wb"
            if mode == "wb":
                resume_pos = 0
            with open(dest, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {url} -> {dest}")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
