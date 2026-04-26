import argparse
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bert_sentiment_model import tokenize_sentiment_data
from berkeley_model import WeightedMSETrainer, make_hf_dataset, tokenize


def filter_english(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:

    try:
        from langdetect import detect

        def is_english(text):
            try:
                return detect(text) == "en"
            except Exception:
                return False

        df = df[df[text_col].apply(is_english)]
    except ImportError:
        print("langdetect not installed — skipping language filter.")

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows after language filter: {len(df)}")
    return df



def predict_sentiment(df: pd.DataFrame, model_path: str, text_col: str = "text") -> pd.DataFrame:
    """Add 'predicted_sentiment' and 'sentiment_labels' columns to df."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    enc = tokenize_sentiment_data(df[text_col], tokenizer)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    })

    from transformers import Trainer, TrainingArguments
    dummy_args = TrainingArguments(output_dir="/tmp/sentiment_infer", no_cuda=not torch.cuda.is_available())
    trainer = Trainer(model=model, args=dummy_args)

    preds_out = trainer.predict(ds)
    logits = preds_out.predictions
    values = 1 / (1 + np.exp(-logits))  # sigmoid

    df["predicted_sentiment"] = values.flatten()
    df["sentiment_labels"] = pd.cut(
        df["predicted_sentiment"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["very negative", "negative", "neutral", "positive", "very positive"],
    )
    print("Sentiment predictions added.")
    return df


def predict_berkeley(df: pd.DataFrame, model_path: str, text_col: str = "text") -> pd.DataFrame:
    """Add predicted Berkeley toxicity columns to df."""
    with open(f"{model_path}/labels.json") as f:
        label_cols = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    enc = tokenize(df[text_col].tolist(), tokenizer)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    })

    trainer = WeightedMSETrainer(model=model)
    preds_out = trainer.predict(ds)
    values = torch.sigmoid(torch.tensor(preds_out.predictions)).numpy()

    for i, col in enumerate(label_cols):
        df[f"predicted_berkley_{col}"] = values[:, i]

    print("Berkeley toxicity predictions added.")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tweets.csv")
    parser.add_argument("--output", default="tweets_with_scores.csv")
    parser.add_argument("--sentiment_model_path", default="./sentiment_roberta_model_final")
    parser.add_argument("--berkeley_model_path", default="./uc_berkeley_model_best")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = filter_english(df)
    df = predict_sentiment(df, args.sentiment_model_path)
    df = predict_berkeley(df, args.berkeley_model_path)
    df.to_csv(args.output, index=False)
    print(f"Saved enriched DataFrame to {args.output}")


if __name__ == "__main__":
    main()