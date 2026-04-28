"""
predict.py — Prediction Utilities for Spam Detection
=====================================================

This module provides functions to load the saved model and vectorizer
from disk, and to classify new email / SMS messages as Spam or Not Spam.

It is designed to be imported by  main.py  (the CLI interface) or by any
other script that needs to make predictions after the model has been trained.
"""

import pickle
from pathlib import Path
from typing import Tuple

from src.preprocessing import clean_text

# ---------------------------------------------------------------------------
# Path constants (same as in train.py for consistency)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def load_model() -> Tuple:
    """
    Load the trained Naive Bayes model and TF-IDF vectorizer from disk.

    Returns
    -------
    tuple
        (model, vectorizer) — both are unpickled scikit-learn objects.

    Raises
    ------
    FileNotFoundError
        If the model or vectorizer pickle files do not exist.
        This usually means you need to run the training script first.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}.\n"
            "Please run the training script first:\n"
            "  python -m src.train"
        )
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {VECTORIZER_PATH}.\n"
            "Please run the training script first:\n"
            "  python -m src.train"
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_message(text: str, model=None, vectorizer=None) -> dict:
    """
    Classify a single email / SMS message as Spam or Not Spam.

    Parameters
    ----------
    text : str
        The raw, unprocessed message text.
    model : sklearn estimator, optional
        A pre-loaded model.  If not provided, the function will load the
        saved model from disk automatically.
    vectorizer : TfidfVectorizer, optional
        A pre-loaded vectorizer.  If not provided, it will be loaded.

    Returns
    -------
    dict
        {
            "original_text":  the raw input,
            "cleaned_text":   the preprocessed version,
            "prediction":     "Spam" or "Not Spam",
            "label":          1 (spam) or 0 (ham),
            "confidence":     probability of the predicted class (0-1),
        }

    Raises
    ------
    ValueError
        If the input text is empty or contains only whitespace.
    FileNotFoundError
        If model files are missing (see load_model()).
    """
    # --- Input Validation ---------------------------------------------------
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    # --- Load model if not provided ----------------------------------------
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    # --- Preprocess the text ------------------------------------------------
    cleaned = clean_text(text)

    # --- Vectorize ----------------------------------------------------------
    # transform() expects an iterable of documents, so we wrap in a list.
    text_vectorized = vectorizer.transform([cleaned])

    # --- Predict ------------------------------------------------------------
    prediction = model.predict(text_vectorized)[0]            # 0 or 1
    probabilities = model.predict_proba(text_vectorized)[0]   # [P(ham), P(spam)]

    label = "Spam" if prediction == 1 else "Not Spam"
    confidence = probabilities[prediction]  # probability of the chosen class

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "prediction": label,
        "label": int(prediction),
        "confidence": round(float(confidence), 4),
    }
