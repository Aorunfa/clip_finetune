from torch import distributed as dist
import os
from contextlib import contextmanager
import torch.nn as nn
import torch
import random
from datetime import timedelta
import numpy as np

RANK = int(os.environ.get('RANK', -1))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

def is_master():
    if RANK in [0, -1]:
        return True
    return False

def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensures all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank])

def setup_ddp_envs(seed=123, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.cuda.set_device(RANK)
    

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)  # warn if deterministic is not possible
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    dist.init_process_group(
        backend="nccl" if dist.is_nccl_available() else "gloo",
        timeout=timedelta(seconds=10800),  # 3 hours
        rank=RANK,
        world_size=WORLD_SIZE,
    )