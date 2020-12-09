from os import terminal_size
from typing import List
import re
from functools import lru_cache
import unicodedata

SPACE_NORMALIZER = re.compile(r"\s+")
# PUNCTUATION_PATTERN = re.compile(r"([\[\]\"'“”(),.!?@#$%&+\-–:;…])")


def tokenize(sent) -> List[str]:
    sent = SPACE_NORMALIZER.sub(" ", sent)
    sent = sent.strip()
    return sent.split()


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


# def better_tokenize(sent) -> List[str]:
#     return re.split(r"\s+", PUNCTUATION_PATTERN.sub(r" \1 ", sent).strip())


def better_tokenize(text, lower=False):
    if lower:
        text = text.lower()
    start_new_word = True
    output = []
    for char in text:
        if _is_chinese_char(char):
            continue
        elif _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
            
    ret = ["".join(x) for x in output]
    return ret


@lru_cache(maxsize=None)
def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


@lru_cache(maxsize=None)
def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False
