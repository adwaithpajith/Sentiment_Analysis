"""
preprocess.py
-------------
Minimal text preprocessing for SBERT-based pipeline.

IMPORTANT DESIGN DECISION:
    SBERT (Sentence-BERT) was trained on raw, natural language sentences.
    Aggressive preprocessing — stemming, stopword removal, lowercasing —
    DESTROYS the semantic signal SBERT was trained to encode.

    For example:
        "not amazing"  →  after stemming/stopword removal  →  "amaz"
        SBERT correctly embeds "not amazing" as semantically negative.
        The preprocessed version loses the negation entirely.

    Therefore: preprocessing is limited to whitespace normalization
    and basic Unicode cleaning only. SBERT handles the rest.
"""

import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Minimal cleaning:
      - Strip leading/trailing whitespace
      - Collapse internal whitespace runs
      - Remove non-printable control characters
      - Normalize smart quotes / apostrophes
    """
    text = text.strip()
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)          # control chars
    text = re.sub(r'[\u2018\u2019]', "'", text)            # smart quotes
    text = re.sub(r'[\u201C\u201D]', '"', text)            # smart double quotes
    text = re.sub(r'\s+', ' ', text)                       # collapse whitespace
    return text


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load CSV and apply minimal cleaning.
    Expected columns: review, label
    Returns DataFrame with added 'clean_review' column.
    """
    df = pd.read_csv(filepath)

    required_cols = {'review', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}. "
                         f"Found: {set(df.columns)}")

    df = df.dropna(subset=['review', 'label']).reset_index(drop=True)
    df['clean_review'] = df['review'].apply(clean_text)

    label_dist = df['label'].value_counts().to_dict()
    print(f"✅ Loaded {len(df)} reviews | Distribution: {label_dist}")
    return df


if __name__ == "__main__":
    df = load_and_preprocess("data/reviews.csv")
    print(df[['review', 'clean_review']].head(5))
