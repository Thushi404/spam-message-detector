"""
preprocessing.py — Text Preprocessing Utilities for Spam Detection
===================================================================

This module contains all the NLP preprocessing functions used to clean
and prepare raw email text before it can be fed into a machine learning model.

Pipeline:
  1. Convert text to lowercase
  2. Remove punctuation and special characters
  3. Tokenize the text into individual words
  4. Remove common English stopwords (e.g., "the", "is", "and")
  5. Apply stemming to reduce words to their root form

Why preprocess?
  Raw text contains noise (punctuation, casing differences, common filler
  words) that adds no useful signal for classification. Preprocessing
  strips away that noise so the model can focus on the words that truly
  distinguish spam from legitimate email.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------
# One-time NLTK resource downloads
# ---------------------------------------------------------------------------
# NLTK ships as a lightweight package; actual language data (word lists,
# tokenizers, etc.) must be downloaded separately.  We only need two:
#   • 'punkt_tab'  — a pre-trained sentence/word tokenizer
#   • 'stopwords'  — lists of common "filler" words for many languages
# The `quiet=True` flag suppresses console output if they're already present.
# ---------------------------------------------------------------------------
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Reusable objects created once and shared across all function calls
# ---------------------------------------------------------------------------
# PorterStemmer is a rule-based algorithm that chops word suffixes to
# produce a "stem".  For example:
#   "running" → "run",  "connected" → "connect",  "fishing" → "fish"
# This helps the model treat different forms of the same word as identical.
stemmer = PorterStemmer()

# A frozen set of English stopwords for O(1) membership testing.
# Examples: {"the", "is", "at", "which", "on", ...}
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Apply the full NLP preprocessing pipeline to a single piece of text.

    Steps performed (in order):
      1. Lowercase the entire string so "FREE" and "free" are treated equally.
      2. Strip out every character that is not a letter or a space.
         This removes digits, punctuation, and special symbols.
      3. Split the cleaned string into a list of individual words (tokens).
      4. Remove stopwords — common English words that carry little meaning.
      5. Stem each remaining word down to its root form.
      6. Re-join the stemmed tokens into a single space-separated string.

    Parameters
    ----------
    text : str
        The raw email body or subject line.

    Returns
    -------
    str
        The fully cleaned and stemmed text, ready for vectorization.

    Example
    -------
    >>> clean_text("CONGRATULATIONS!!! You have WON a FREE ticket!!!")
    'congratul free ticket'
    """
    # --- Step 1: Lowercase ---------------------------------------------------
    # "FREE" and "free" should be the same token to the model.
    text = text.lower()

    # --- Step 2: Remove non-alphabetic characters ----------------------------
    # We keep only letters (a-z) and spaces.  Everything else (digits,
    # punctuation, emojis, HTML tags, etc.) is replaced with a space.
    text = re.sub(r"[^a-z\s]", " ", text)

    # --- Step 3: Tokenize (split into words) ---------------------------------
    # str.split() splits on any whitespace and automatically ignores
    # leading/trailing spaces and multiple consecutive spaces.
    words = text.split()

    # --- Step 4 & 5: Remove stopwords + Stem ---------------------------------
    # We combine these two steps into one list comprehension for efficiency:
    #   • Skip the word if it appears in the STOP_WORDS set.
    #   • Otherwise, apply the PorterStemmer to reduce it to its root.
    processed_words = [
        stemmer.stem(word)
        for word in words
        if word not in STOP_WORDS
    ]

    # --- Step 6: Rejoin into a single string ---------------------------------
    return " ".join(processed_words)
