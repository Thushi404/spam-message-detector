"""
train.py — Model Training & Evaluation for Spam Detection
==========================================================

This script handles the complete machine learning workflow:
  1. Load the dataset (SMS Spam Collection from UCI / bundled CSV)
  2. Preprocess every message using our NLP pipeline
  3. Convert cleaned text into numerical feature vectors (TF-IDF)
  4. Split the data into training (80%) and testing (20%) sets
  5. Train a Multinomial Naive Bayes classifier
  6. Evaluate the model and display metrics + visualizations
  7. Save the trained model and vectorizer to disk with pickle

Why Multinomial Naive Bayes?
  It is fast, works well with word-frequency features (like TF-IDF),
  and is a proven baseline for text classification tasks — especially
  spam detection.  Despite its simplicity, it often delivers
  surprisingly competitive accuracy.
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# Import our custom preprocessing function from the src package
from src.preprocessing import clean_text

# Use a non-interactive backend so plots can be saved even without a display
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# All paths are relative to the project root (one level above this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

# File where the trained model + vectorizer will be persisted
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# ==============================
# STEP 1 — Load the Dataset
# ==============================
def load_dataset() -> pd.DataFrame:
    """
    Load the SMS Spam Collection dataset.

    The function first looks for a local CSV file at  data/spam.csv .
    If none is found, it automatically downloads the well-known
    "SMS Spam Collection" dataset from the UCI Machine Learning Repository.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
          • 'label'   — "ham" (not spam) or "spam"
          • 'message' — the raw email / SMS text
    """
    csv_path = DATA_DIR / "spam.csv"

    if csv_path.exists():
        # ------------------------------------------------------------------
        # Try loading the local CSV.  Many versions of this dataset use
        # 'latin-1' (ISO-8859-1) encoding because messages may contain
        # accented characters.
        # ------------------------------------------------------------------
        print(f"[INFO] Loading dataset from {csv_path}")
        try:
            # Format 1: CSV with columns named v1 (label) and v2 (message)
            df = pd.read_csv(csv_path, encoding="latin-1")
            if "v1" in df.columns and "v2" in df.columns:
                df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
            elif "label" in df.columns and "message" in df.columns:
                df = df[["label", "message"]]
            else:
                # Assume first two columns are label and message
                df = df.iloc[:, :2]
                df.columns = ["label", "message"]
        except Exception as e:
            print(f"[WARNING] Could not parse {csv_path}: {e}")
            print("[INFO] Falling back to built-in sample dataset.")
            df = _create_sample_dataset()
    else:
        # ------------------------------------------------------------------
        # No local file → generate a small built-in sample so the project
        # works right out of the box without any external downloads.
        # ------------------------------------------------------------------
        print("[INFO] No dataset found at data/spam.csv")
        print("[INFO] Generating built-in sample dataset for demonstration...")
        df = _create_sample_dataset()

    # Drop any rows that have missing values in the columns we need
    df.dropna(subset=["label", "message"], inplace=True)

    # Standardize labels to lowercase for consistency
    df["label"] = df["label"].str.lower().str.strip()

    print(f"[INFO] Dataset loaded — {len(df)} messages "
          f"({(df['label'] == 'spam').sum()} spam, "
          f"{(df['label'] == 'ham').sum()} ham)")
    return df


def _create_sample_dataset() -> pd.DataFrame:
    """
    Create a small but realistic sample dataset for demonstration.

    This dataset contains 40 example messages (20 spam + 20 ham) that
    capture the typical linguistic patterns of each class so the model
    can learn meaningful decision boundaries even on this tiny corpus.

    In a real project you would replace this with a large, externally
    sourced dataset (e.g., the full SMS Spam Collection — 5,574 messages).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    spam_messages = [
        "CONGRATULATIONS! You've been selected to receive a FREE iPhone! Click here NOW!",
        "WINNER!! You have been chosen to WIN a $1000 Walmart gift card. Call now!",
        "Urgent! Your account has been compromised. Verify your identity immediately.",
        "FREE FREE FREE! Buy one get one free on all products today only!!!",
        "You have won $5000! Send your bank details to claim your prize money now.",
        "SPECIAL OFFER: Get 90% discount on luxury watches. Limited time only!",
        "Congratulations! You are our lucky winner today. Reply YES to claim reward.",
        "ALERT: Suspicious activity detected on your account. Click link to verify.",
        "Make money fast! Work from home and earn $5000 per week guaranteed!",
        "Your loan has been approved! Get cash immediately with no credit check.",
        "FLASH SALE! Everything must go! 95% off all items. Order now before sold out!",
        "You have been pre-approved for a credit card with $10000 limit. Apply now!",
        "WARNING: Your computer is infected with virus! Download our free antivirus NOW!",
        "Earn extra income! Join our network marketing team and become rich overnight!",
        "FREE entry to our exclusive VIP club! Text JOIN to 80808 right now!",
        "Your mobile phone has won a prize of $2500! Call this number to claim it!",
        "Diet pills that actually work! Lose 30 pounds in 30 days guaranteed or money back!",
        "Secret investment opportunity! Double your money in just 7 days. Act now!",
        "You're eligible for a government grant of $25000! No repayment required. Apply!",
        "Last chance! Your subscription is about to expire. Renew now for 80% discount!",
    ]

    ham_messages = [
        "Hey, are we still meeting for lunch tomorrow at noon?",
        "Just wanted to let you know the meeting has been rescheduled to 3 PM.",
        "Can you pick up some groceries on your way home? We need milk and bread.",
        "Thanks for sending the report. I'll review it and get back to you by Friday.",
        "Happy birthday! Hope you have an amazing day with family and friends.",
        "The project deadline has been extended by one week. No rush on your part.",
        "I'll be running a bit late today. Traffic is really bad this morning.",
        "Did you see the email from the manager? We need to update the presentation.",
        "Let me know when you're free to discuss the quarterly review results.",
        "The kids have a school play next Thursday. Can you attend?",
        "I've attached the meeting notes from yesterday's call. Please review.",
        "Looking forward to the team dinner tonight! See you there at seven.",
        "Could you please send me the updated budget spreadsheet when you get a chance?",
        "The doctor's appointment is confirmed for next Monday at 10:30 AM.",
        "Great job on the presentation today! The client was really impressed.",
        "Reminder: Please submit your timesheet before end of day Friday.",
        "I finished reading the book you recommended. It was really insightful!",
        "Can we reschedule our one-on-one to Wednesday instead of Tuesday?",
        "The weather forecast says rain all week. Don't forget your umbrella!",
        "Just checking in — how's the new project going? Need any help?",
    ]

    # Build the DataFrame
    data = {
        "label": ["spam"] * len(spam_messages) + ["ham"] * len(ham_messages),
        "message": spam_messages + ham_messages,
    }
    df = pd.DataFrame(data)

    # Save to CSV so it's available for future runs
    csv_path = DATA_DIR / "spam.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Sample dataset saved to {csv_path}")

    return df


# ==============================
# STEP 2 — Preprocess the Data
# ==============================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the NLP cleaning pipeline to every message in the DataFrame.

    A new column 'cleaned_message' is added that contains the preprocessed
    version of each message (lowercased, no punctuation, no stopwords,
    stemmed).

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'message' column with raw text.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional 'cleaned_message' column.
    """
    print("[INFO] Preprocessing text data...")
    df = df.copy()  # avoid mutating the caller's DataFrame
    df["cleaned_message"] = df["message"].apply(clean_text)
    print("[INFO] Preprocessing complete.")
    return df


# =============================================
# STEP 3 — Vectorize + Split + Train + Evaluate
# =============================================
def train_and_evaluate() -> None:
    """
    Execute the entire training pipeline end-to-end:
      1. Load data
      2. Preprocess
      3. Vectorize (TF-IDF)
      4. Train / test split
      5. Train Multinomial Naive Bayes
      6. Evaluate and display results
      7. Save model & vectorizer to disk

    This function is designed to be called as the main entry point of the
    training script (see ``if __name__ == "__main__"`` block at the bottom).
    """
    # Ensure output directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load ----------------------------------------------------------
    df = load_dataset()

    # ---- 2. Preprocess ----------------------------------------------------
    df = preprocess_data(df)

    # Convert labels to binary: spam → 1, ham → 0
    # This makes it easier for the model and metrics functions.
    df["label_encoded"] = df["label"].map({"spam": 1, "ham": 0})

    # Show a preview so we can sanity-check the pipeline
    print("\n[INFO] Sample preprocessed data:")
    print(df[["label", "message", "cleaned_message"]].head(5).to_string(index=False))
    print()

    # ---- 3. TF-IDF Vectorization ------------------------------------------
    # TF-IDF stands for "Term Frequency – Inverse Document Frequency".
    #
    # Term Frequency (TF):
    #   How often a word appears in a single document.
    #   Words that appear many times in a message get a higher score.
    #
    # Inverse Document Frequency (IDF):
    #   Penalizes words that appear in many documents (they're less useful
    #   for distinguishing classes).  Rare words get a higher score.
    #
    # TF-IDF = TF × IDF
    #   Words that are frequent locally but rare globally receive the
    #   highest weight — exactly the kind of words that help classify
    #   spam vs. ham.
    #
    # max_features=5000 limits the vocabulary to the top 5,000 words
    # to keep the feature matrix manageable.
    print("[INFO] Vectorizing text with TF-IDF (max 5000 features)...")
    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df["cleaned_message"])  # sparse matrix
    y = df["label_encoded"]

    print(f"[INFO] Feature matrix shape: {X.shape}")

    # ---- 4. Train / Test Split --------------------------------------------
    # 80% of the data is used for training, 20% for testing.
    # random_state=42 ensures reproducibility — you'll get the same split
    # every time you run this script.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Testing  set: {X_test.shape[0]} samples")

    # ---- 5. Train the Model -----------------------------------------------
    # Multinomial Naive Bayes is a probabilistic classifier well-suited
    # for text classification with word-count / TF-IDF features.
    #
    # How it works (simplified):
    #   For each class (spam / ham), it learns the probability of every
    #   word appearing.  Given a new message, it multiplies the learned
    #   probabilities of all its words for each class and picks the class
    #   with the highest overall probability.
    print("[INFO] Training Multinomial Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing parameter
    model.fit(X_train, y_train)
    print("[INFO] Training complete!")

    # ---- 6. Evaluate the Model --------------------------------------------
    y_pred = model.predict(X_test)

    # --- Core Metrics ---
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("            MODEL EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 55)

    # --- Detailed Classification Report ---
    # Shows precision, recall, and F1 for EACH class individually.
    print("\n[INFO] Detailed Classification Report:")
    report = classification_report(
        y_test, y_pred, target_names=["Ham (Not Spam)", "Spam"]
    )
    print(report)

    # --- Confusion Matrix ---
    # A confusion matrix shows:
    #   • True Positives  (TP): correctly predicted spam
    #   • True Negatives  (TN): correctly predicted ham
    #   • False Positives (FP): ham wrongly flagged as spam
    #   • False Negatives (FN): spam that slipped through as ham
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm)

    # --- Dataset Distribution Plot ---
    _plot_label_distribution(df)

    # ---- 7. Save Model & Vectorizer to Disk --------------------------------
    _save_artifacts(model, vectorizer)

    print("\n[INFO] All done! You can now run  main.py  to classify emails.")


def _plot_confusion_matrix(cm: np.ndarray) -> None:
    """
    Generate and save a heatmap visualization of the confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        A 2×2 confusion matrix from sklearn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,               # show numbers inside each cell
        fmt="d",                   # format numbers as integers
        cmap="Blues",              # blue color gradient
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("Actual Label", fontsize=13)
    plt.tight_layout()

    save_path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to {save_path}")


def _plot_label_distribution(df: pd.DataFrame) -> None:
    """
    Generate and save a bar chart showing how many spam vs. ham messages
    exist in the dataset.  This helps identify class imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'label' column.
    """
    plt.figure(figsize=(7, 5))
    colors = ["#4CAF50", "#F44336"]  # green for ham, red for spam
    counts = df["label"].value_counts()
    ax = counts.plot(kind="bar", color=colors, edgecolor="white", linewidth=1.2)

    # Add count labels on top of each bar
    for i, (label, count) in enumerate(counts.items()):
        ax.text(i, count + 0.5, str(count), ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    plt.title("Dataset Label Distribution", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Label", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.xticks(rotation=0)
    plt.tight_layout()

    save_path = PLOTS_DIR / "label_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Label distribution plot saved to {save_path}")


def _save_artifacts(model: MultinomialNB, vectorizer: TfidfVectorizer) -> None:
    """
    Persist the trained model and the fitted TF-IDF vectorizer using pickle.

    Why save both?
      The vectorizer learned a specific vocabulary and IDF weights during
      training.  At prediction time we need the *exact same* vectorizer
      to transform new text into the same feature space the model expects.
      If we used a freshly created vectorizer, the column indices would
      not match and predictions would be meaningless.

    Parameters
    ----------
    model : MultinomialNB
        The trained classifier.
    vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer.
    """
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectorizer saved to {VECTORIZER_PATH}")


# ---------------------------------------------------------------------------
# Allow this module to be run standalone:  python -m src.train
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_evaluate()
