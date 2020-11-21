import random
from typing import List
import copy
import importlib_resources
import pickle
from .defs import AugOp
import json
from ..util import load_resource


class CharRandomSwap(AugOp):
    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = random.randint(0, len(ret) - 1)
            word = ret[pos]
            if len(word) > 4:
                cpos = random.randint(1, len(word) - 2)
                word = word[:cpos] \
                        + word[cpos + 1] + word[cpos] \
                        + word[cpos + 2:]
                ret[pos] = word
        return ret


class CharRandomDelete(AugOp):
    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = random.randint(0, len(ret) - 1)
            word = ret[pos]
            if len(word) > 2:
                cpos = random.randint(0, len(word) - 2)
                word = word[:cpos + 1] + word[cpos + 2:]
                ret[pos] = word
        return ret


class CharTypoSub(AugOp):
    def __init__(self, aug_p: float) -> None:
        super().__init__(aug_p)
        a = json.load(open(load_resource("typo_ocr.json")))
        b = json.load(open(load_resource("typo_keyboard.json")))
        self.mapping = {}
        for k in set(a.keys()).union(set(b.keys())):
            vals = []
            if k in a:
                vals.extend(a[k])
            if k in b:
                vals.extend(b[k])
            self.mapping[k] = list(set(vals))

    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = random.randint(0, len(ret) - 1)
            word = ret[pos]
            cpos = random.randint(1, len(word) - 2)
            if word[cpos] in self.mapping:
                word = word[:cpos] + random.choice(
                    self.mapping[word[cpos]]) + word[cpos + 1:]
            ret[pos] = word
        return ret
