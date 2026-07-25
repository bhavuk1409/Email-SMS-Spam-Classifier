# Spam Classifier — Email & SMS

A Streamlit app that classifies a message as **Spam** or **Not Spam** using
classic NLP preprocessing and a TF-IDF + machine learning pipeline trained
on labeled SMS/email data.

![python](https://img.shields.io/badge/python-3.7%2B-3776ab) ![ui](https://img.shields.io/badge/UI-Streamlit-ff4b4b) ![nlp](https://img.shields.io/badge/NLP-NLTK-154f5b) ![ml](https://img.shields.io/badge/ML-scikit--learn-f7931e)

---

## What it does

1. **Enter a message** — paste any email or SMS text into the app.
2. **Text preprocessing** — the message is:
   - Lowercased
   - Tokenized (`nltk.word_tokenize`)
   - Stripped of non-alphanumeric tokens
   - Stripped of English stopwords and punctuation
   - Stemmed with NLTK's `PorterStemmer`
3. **Vectorization** — the cleaned text is transformed with a pretrained
   **TF-IDF vectorizer** (`vectorizer.pkl`).
4. **Prediction** — a pretrained classifier (`model.pkl`) predicts
   **Spam** or **Not Spam** from the TF-IDF features.

The model was trained and evaluated in `sms-spam-detection.ipynb` on the
labeled dataset in `spam-3.csv`, then exported with `pickle` for the
Streamlit app to load directly — no retraining needed to run the app.

---

## Repository layout

```
├── app.py                     # Streamlit app: input, preprocessing, prediction, display
├── sms-spam-detection.ipynb    # Notebook: EDA, preprocessing, model training/evaluation
├── spam-3.csv                   # Labeled training dataset
├── vectorizer.pkl                 # Pretrained TF-IDF vectorizer
└── model.pkl                       # Pretrained classifier
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/bhavuk1409/Email-SMS-Spam-Classifier.git
cd Email-SMS-Spam-Classifier

pip install pandas numpy scikit-learn nltk streamlit
```

### 2. Download NLTK data

The app needs NLTK's tokenizer and stopword list:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste a message, click **Predict**, and
see whether it's classified as Spam or Not Spam.

---

## Retraining

To retrain on new data or experiment with the pipeline, open
`sms-spam-detection.ipynb`. It covers exploratory data analysis, the same
text-preprocessing steps used in `app.py`, TF-IDF feature extraction, and
model training/evaluation on `spam-3.csv`. Re-export `vectorizer.pkl` and
`model.pkl` after retraining to update what `app.py` loads.

---

## Tech stack

- **Streamlit** — UI
- **NLTK** — tokenization, stopword removal, stemming
- **scikit-learn** — TF-IDF vectorization and the classifier
- **pandas / numpy** — data handling in the training notebook

---

## License

MIT
