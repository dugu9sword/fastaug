from typing import List
import re

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize(sent) -> List[str]:
    sent = SPACE_NORMALIZER.sub(" ", sent)
    sent = sent.strip()
    return sent.split()


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)