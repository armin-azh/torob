from constants import *
import re

invalid_reg_text = '[^{}]'.format("".join(valid_chars))
invalid_reg = re.compile(invalid_reg_text)


def normalize_text(text, to_lower=True, remove_invalid=True):
    text = text.strip()
    text = text.translate(translation_table)
    if to_lower:
        text = text.lower()
    if remove_invalid:
        text = invalid_reg.sub(' ', text)
    # Replace consecutive whitespaces with a single space character.
    text = re.sub(r"\s+", " ", text)
    return text
