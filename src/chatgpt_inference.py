import argparse
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from bert_sentiment_model import tokenize_sentiment_data
from berkeley_model import WeightedMSETrainer, make_hf_dataset, tokenize



def load_chatgpt_data(sample: int = 200_000) -> pd.DataFrame:

    ds = load_dataset("lmsys/lmsys-chat-1m")
    df = ds["train"].to_pandas()
    df = df[df["language"] == "English"]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.iloc[:sample]

    def get_user_messages(conversation_list):
        msgs = [t["content"] for t in conversation_list if t["role"] == "user"]
        return "\n".join(msgs) if msgs else None

    df["anonymous_user_message"] = df["conversation"].apply(get_user_messages)
    df = df.dropna(subset=["anonymous_user_message"]).reset_index(drop=True)
    print(f"ChatGPT anonymous messages: {len(df)}")
    return df



def predict_sentiment(df: pd.DataFrame, model_path: str) -> pd.DataFrame:

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    enc = tokenize_sentiment_data(df["anonymous_user_message"], tokenizer)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    })

    dummy_args = TrainingArguments(
        output_dir="/tmp/chatgpt_sentiment_infer",
        no_cuda=not torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=dummy_args)
    preds_out = trainer.predict(ds)
    logits = preds_out.predictions
    values = 1 / (1 + np.exp(-logits))

    df["predicted_sentiment"] = values.flatten()
    df["sentiment_labels"] = pd.cut(
        df["predicted_sentiment"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["very negative", "negative", "neutral", "positive", "very positive"],
    )
    print("Sentiment predictions added to ChatGPT data.")
    return df


def predict_berkeley(df: pd.DataFrame, model_path: str) -> pd.DataFrame:

    with open(f"{model_path}/labels.json") as f:
        label_cols = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    enc = tokenize(df["anonymous_user_message"].tolist(), tokenizer)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    })

    trainer = WeightedMSETrainer(model=model)
    preds_out = trainer.predict(ds)
    values = torch.sigmoid(torch.tensor(preds_out.predictions)).numpy()

    for i, col in enumerate(label_cols):
        df[f"predicted_berkley_{col}"] = values[:, i]

    print("Berkeley toxicity predictions added to ChatGPT data.")
    return df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="chatgpt_anonymous_df.csv")
    parser.add_argument("--sentiment_model_path", default="./sentiment_roberta_model_final")
    parser.add_argument("--berkeley_model_path", default="./uc_berkeley_model_best")
    parser.add_argument("--sample", type=int, default=200_000)
    args = parser.parse_args()

    df = load_chatgpt_data(sample=args.sample)
    df = predict_sentiment(df, args.sentiment_model_path)
    df = predict_berkeley(df, args.berkeley_model_path)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()