# utils/text_utils.py

import re
import string

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)       # remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # remove punctuation/numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
