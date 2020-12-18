import random
from typing import List, Dict
import copy
from .defs import AugOp
from functools import lru_cache
from nltk.corpus import wordnet as wn
from ..util import load_resource, randint
import json

# extracted from nltk
STOPWORDS = {
    "ourselves", "hers", "between", "yourself", "but", "again", "there",
    "about", "once", "during", "out", "very", "having", "with", "they", "own",
    "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of",
    "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as",
    "from", "him", "each", "the", "themselves", "until", "below", "are", "we",
    "these", "your", "his", "through", "don", "nor", "me", "were", "her",
    "more", "himself", "this", "down", "should", "our", "their", "while",
    "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when",
    "at", "any", "before", "them", "same", "and", "been", "have", "in", "will",
    "on", "does", "yourselves", "then", "that", "because", "what", "over",
    "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself",
    "has", "just", "where", "too", "only", "myself", "which", "those", "i",
    "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a",
    "by", "doing", "it", "how", "further", "was", "here", "than"
}


class WordRandomMask(AugOp):
    def __init__(self, aug_p: float, mask: str = '_'):
        super().__init__(aug_p)
        self.mask = mask

    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        tk_len = len(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = randint(0, tk_len)
            ret[pos] = self.mask
        return ret


class WordRandomSwap(AugOp):
    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        tk_len = len(tokens)
        if tk_len > 2:
            for _ in range(self.aug_len(tokens)):
                pos = randint(0, tk_len - 1)
                ret[pos], ret[pos + 1] = ret[pos + 1], ret[pos]
        return ret


class WordRandomDelete(AugOp):
    def __call__(self, tokens: List[str]) -> List[str]:
        ret = copy.copy(tokens)
        for _ in range(self.aug_len(tokens)):
            pos = randint(0, len(ret))
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
                #  and tokens[idx] not in STOPWORDS:
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
        self.morphs = json.load(open(load_resource("morphs.json"), "r"))

    @lru_cache(maxsize=None)
    def has_cands(self, word):
        return word in self.morphs

    @lru_cache(maxsize=None)
    def get_cands(self, word):
        return self.morphs[word]


# class WordEmbedSub(WordSub):
#     def __init__(self, aug_p: float) -> None:
#         super().__init__(aug_p)
#         embed_path = DownloadUtil.download_counter_fitting_if_not_exists()
#         df = pandas.read_csv(embed_path, sep=" ", header=None)
#         words = list(map(str, df.values[:, 0]))
#         vecs = torch.tensor(df.values[:, 1:].astype(np.float32))
#         self._word2idx = {words[i]: i for i in range(len(words))}
#         self._idx2word = {i: words[i] for i in range(len(words))}
#         self.embed_util = EmbeddingNbrUtil(
#             vecs,
#             word2idx=self._word2idx,
#             idx2word=self._idx2word,
#         )

#     @lru_cache(maxsize=None)
#     def has_cands(self, word):
#         return word in self._word2idx

#     @lru_cache(maxsize=None)
#     def get_cands(self, word):
#         return self.embed_util.find_neighbours(
#             word,
#             measure='cos',  #
#             topk=10,  #
#             return_words=True  #
#         )

# class WordEmbedSub(WordSub):
#     def __init__(self, aug_p: float) -> None:
#         super().__init__(aug_p)
#         self.cands = json.load(
#             open(load_resource("embed_top_16_dist_dot25.json"), "r"))

#     @lru_cache(maxsize=None)
#     def has_cands(self, word):
#         return word in self.cands

#     @lru_cache(maxsize=None)
#     def get_cands(self, word):
#         return self.cands[word]


class WordDictSub(WordSub):
    def __init__(self, aug_p: float, cands: Dict[str, List[str]]) -> None:
        super().__init__(aug_p)
        self.cands = cands

    @lru_cache(maxsize=None)
    def has_cands(self, word):
        return word in self.cands

    @lru_cache(maxsize=None)
    def get_cands(self, word):
        return self.cands[word]


class WordEmbedSub(WordDictSub):
    def __init__(self, aug_p: float) -> None:
        super().__init__(aug_p, cands=None)
        self.cands = json.load(
            open(load_resource("embed_top_16_dist_dot25.json"), "r"))
