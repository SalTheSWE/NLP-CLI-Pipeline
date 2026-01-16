import pandas as pd
import re
def word_counts(series:pd.Series):
    counts = []
    for e in series:
        words = e.split(" ")
        counts.append(len(words))
    return counts
def char_counts(series:pd.Series):
    counts = []
    for e in series:
        words = e.split(" ")
        chars = 0
        for word in words:
            chars+=len(word)
            counts.append(chars)
    return counts
def clean_arabic_text(text):
    
    ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
    TATWEEL = "\u0640"

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#\w+")
    ARABIC_PUNCT = re.compile(r"[،؛؟«»ـ…]")
    if pd.isna(text):
        return ""

    text = str(text)

    # remove links + mentions + hashtags
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub("", text)

    # remove tashkeel (diacritics)
    text = ARABIC_DIACRITICS.sub("", text)

    # remove tatweel
    text = text.replace(TATWEEL, "")

    # remove Arabic punctuation (tarqeem)
    text = ARABIC_PUNCT.sub(" ", text)
    return text
def remove_stopwords_ar(text: str, stopwords: set) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    tokens = re.split(r"\s+", text)
    tokens = [t for t in tokens if t and t not in stopwords]
    return " ".join(tokens)
ALEF_VARIANTS = re.compile(r"[إأآٱ]")
TATWEEL = "\u0640"
def normalize_arabic(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace(TATWEEL, "")
    text = ALEF_VARIANTS.sub("ا", text)

    # hamza-on-ya / hamza-on-waw (optional but common)
    text = text.replace("ئ", "ي")
    text = text.replace("ؤ", "و")

    # alef maqsoura
    text = text.replace("ى", "ي")

    # taa marbouta (choose one)
    text = text.replace("ة", "ه")  # alternative: "ت"

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
import arabic_reshaper
from bidi.algorithm import get_display
def _fix_arabic(text):
    text = str(text)
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)