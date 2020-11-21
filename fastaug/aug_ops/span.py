import random
from typing import List
import copy
import importlib_resources
import pickle
import numpy as np
from .defs import AugOp


class SpanRandomMask(AugOp):
    def __init__(self, aug_p: float, mask: str = '_'):
        super().__init__(aug_p)
        self.mask = mask

    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        max_mask_len = self.aug_len(tokens)
        total_mask_len = 0
        while total_mask_len < max_mask_len:
            length = len(ret)
            start_idx = random.choice(range(length))
            cur_mask_len = min(
                np.random.geometric(p=0.3, size=1)[0],  # 
                length - start_idx - 1,
                max_mask_len - total_mask_len,
                8)
            total_mask_len += cur_mask_len
            ret = ret[:start_idx] + [self.mask] + ret[start_idx + cur_mask_len:]
        return ret

