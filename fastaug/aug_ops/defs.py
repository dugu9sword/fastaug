from typing import List


class AugOp:
    def __init__(self, aug_p):
        self.aug_p = aug_p

    def aug_len(self, tokens: List[str]):
        return max(1, int(len(tokens) * self.aug_p))

    def __call__(self, tokens: List[str]) -> List[str]:
        raise NotImplementedError