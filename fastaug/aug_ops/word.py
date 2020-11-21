import random
from typing import List
import copy
import importlib_resources
import pickle
from .defs import AugOp
from functools import lru_cache
from nltk.corpus import wordnet as wn
from ..util import load_resource


class WordRandomMask(AugOp):
    def __init__(self, aug_p: float, mask: str = '_'):
        super().__init__(aug_p)
        self.mask = mask

    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        tk_len = len(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = random.randint(0, tk_len - 1)
            ret[pos] = self.mask
        return ret


class WordRandomSwap(AugOp):
    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        tk_len = len(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = random.randint(0, tk_len - 2)
            ret[pos], ret[pos + 1] = ret[pos + 1], ret[pos]
        return ret


class WordRandomDelete(AugOp):
    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = random.randint(0, len(ret) - 1)
            del ret[pos]
        return ret


class WordSub(AugOp):
    def __init__(self, aug_p: float):
        super().__init__(aug_p)

    def has_cands(self, word: str) -> bool:
        raise NotImplementedError

    def get_cands(self, word: str) -> List[str]:
        raise NotImplementedError

    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        all_idxs = list(range(len(tokens)))
        random.shuffle(all_idxs)
        can_sub_idxs = []
        for idx in all_idxs:
            if self.has_cands(tokens[idx]):
                can_sub_idxs.append(idx)
            if len(can_sub_idxs) > self.aug_len(tokens):
                break
        for idx in can_sub_idxs:
            ret[idx] = random.choice(self.get_cands(tokens[idx]))
        return ret


class WordNetSub(WordSub):
    @lru_cache(maxsize=None)
    def has_cands(self, word):
        return len(self.get_cands(word)) != 0

    @lru_cache(maxsize=None)
    def get_cands(self, word, only_unigram=True):
        synonyms = []
        for syn in wn.synsets(word):
            for synonym in syn.lemma_names():
                if "_" in synonym and only_unigram:
                    continue
                else:
                    synonym = synonym.replace("_", " ")
                if synonym in synonyms:
                    continue
                synonyms.append(synonym)
        return synonyms


class WordMorphSub(WordSub):
    def __init__(self, aug_p: float) -> None:
        super().__init__(aug_p)
        self.morphs = pickle.load(open(load_resource("morphs.p"), "rb"))

    @lru_cache(maxsize=None)
    def has_cands(self, word):
        return word in self.morphs

    @lru_cache(maxsize=None)
    def get_cands(self, word):
        return self.morphs[word]
