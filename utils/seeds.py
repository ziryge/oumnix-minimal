import os
import random
from typing import Optional

def set_seed(seed: int = 1337, deterministic: bool = False) -> None:
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
        if deterministic:
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
