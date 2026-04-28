# 📧 Spam Email Detection — Machine Learning Project

A complete, production-quality Python project that classifies emails/SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and Machine Learning.

---

## 📁 Project Structure

```
spam/
├── data/                    # Dataset directory
│   └── spam.csv             # Auto-generated sample (or your own dataset)
├── models/                  # Saved model & vectorizer (created after training)
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── plots/                   # Evaluation visualizations (created after training)
│   ├── confusion_matrix.png
│   └── label_distribution.png
├── src/                     # Source code package
│   ├── __init__.py          # Package initializer
│   ├── preprocessing.py     # NLP text preprocessing functions
│   ├── train.py             # Model training & evaluation pipeline
│   └── predict.py           # Prediction / inference utilities
├── main.py                  # 🚀 CLI entry point
├── requirements.txt         # Python dependencies
└── README.md                # You are here!
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites

- **Python 3.8+** (recommended: Python 3.10+)
- **pip** (comes with Python)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Use Your Own Dataset

Place a CSV file at `data/spam.csv` with the following format:

| label | message                                    |
|-------|--------------------------------------------|
| spam  | CONGRATULATIONS! You won a free iPhone!    |
| ham   | Hey, are we meeting for lunch tomorrow?    |

The script also supports the common `v1` / `v2` column naming convention used by the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset.

If no CSV is found, a built-in sample dataset (40 messages) is automatically generated for demonstration.

---

## 🚀 How to Run

### Option A — Interactive CLI (Recommended)

```bash
python main.py
```

This launches an interactive menu:

| Option | Action                                                |
|--------|-------------------------------------------------------|
| **1**  | Train the model (preprocessing → TF-IDF → Naive Bayes → evaluation) |
| **2**  | Predict — enter any email text and get a Spam / Not Spam verdict     |
| **3**  | Exit                                                  |

### Option B — Train Directly

```bash
python -m src.train
```

---

## 🧠 How It Works

### 1. Text Preprocessing (`src/preprocessing.py`)

Every raw message goes through this pipeline:

| Step | What it does | Example |
|------|-------------|---------|
| Lowercase | Normalizes casing | `"FREE Offer"` → `"free offer"` |
| Remove punctuation | Strips non-letter characters | `"Hello!!!"` → `"Hello"` |
| Remove stopwords | Drops common filler words | `"this is a free offer"` → `"free offer"` |
| Stemming | Reduces words to root form | `"running"` → `"run"` |

### 2. Feature Extraction — TF-IDF

**TF-IDF (Term Frequency – Inverse Document Frequency)** converts each cleaned message into a numerical vector:

- **TF**: How often a word appears in *this* message.
- **IDF**: Penalizes words that appear in *many* messages (they're less discriminative).
- **TF × IDF**: Words that are important locally but rare globally get the highest weight.

### 3. Classification — Multinomial Naive Bayes

A probabilistic model that learns the likelihood of each word appearing in spam vs. ham messages, then uses Bayes' theorem to classify new messages.

### 4. Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **Accuracy** | % of all predictions that were correct |
| **Precision** | Of messages flagged as spam, how many actually were? |
| **Recall** | Of all actual spam, how many did we catch? |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN |

---

## 📊 Output Files

After training (option 1), the following files are generated:

| File | Description |
|------|-------------|
| `models/spam_model.pkl` | Serialized Naive Bayes classifier |
| `models/tfidf_vectorizer.pkl` | Serialized TF-IDF vectorizer |
| `plots/confusion_matrix.png` | Heatmap of the confusion matrix |
| `plots/label_distribution.png` | Bar chart of spam vs. ham distribution |

---

## 📝 Example Prediction

```
📩  Enter message: CONGRATULATIONS! You won a FREE iPhone! Click here NOW!

┌────────────────────── RESULT ──────────────────────┐
│  Prediction : 🚫  SPAM                              │
│  Confidence : 98.5%  ████████████████████            │
│  Cleaned    : congratul free iphon click              │
└────────────────────────────────────────────────────┘
```

---

## 🔧 Technologies Used

| Library | Purpose |
|---------|---------|
| **pandas** | Data loading and manipulation |
| **numpy** | Numerical operations |
| **scikit-learn** | ML model, TF-IDF, metrics, train/test split |
| **matplotlib** | Plotting evaluation charts |
| **seaborn** | Enhanced confusion matrix heatmap |
| **nltk** | Stopwords, stemming, tokenization |

---

## 📌 Key Concepts for Beginners

- **NLP (Natural Language Processing)**: Teaching computers to understand human language.
- **Tokenization**: Splitting text into individual words.
- **Stopwords**: Common words (e.g., "the", "is") that don't help classification.
- **Stemming**: Reducing words to their base/root form (e.g., "fishing" → "fish").
- **TF-IDF**: A way to represent text as numbers that a model can process.
- **Naive Bayes**: A fast, probabilistic classifier based on Bayes' theorem.
- **Confusion Matrix**: A table showing correct vs. incorrect predictions.
- **Pickle**: Python's built-in serialization format for saving/loading objects.

---

## 📄 License

This project is provided for educational purposes. Feel free to use, modify, and distribute.
