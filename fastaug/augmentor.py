from typing import List, Union
from .tokenizer import tokenize, detokenize
from .aug_ops.defs import AugOp
import random


class Augmentor:
    def __init__(self, aug_ops: List[AugOp], pipeline_p: float = 1.0):
        self.aug_ops = aug_ops
        self.pipeline_p = pipeline_p

    def augment(self, sent: str, n: int = 1) -> Union[str, List[str]]:
        ret = []
        for _ in range(n):
            tokens = tokenize(sent)
            for op in self.aug_ops:
                if random.random() < self.pipeline_p:
                    tokens = op(tokens)
            ret.append(detokenize(tokens))
        if n == 1:
            return ret[0]
        else:
            return ret
