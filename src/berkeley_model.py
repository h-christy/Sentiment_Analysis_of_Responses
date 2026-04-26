import json

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from visualize import plot_berkeley_label_means



MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"
DEFAULT_SAVE_PATH = "./uc_berkeley_model_best"

LABEL_COLS = [
    "sentiment", "respect", "insult", "humiliate",
    "status", "dehumanize", "violence", "genocide",
    "attack_defend", "hatespeech",
]



def load_berkeley_data() -> "pd.DataFrame":
    """
    Download the UC Berkeley Measuring Hate Speech dataset from HuggingFace,
    aggregate per comment, and min-max normalise each label column.

    Returns a DataFrame with columns: comment_id, text, <LABEL_COLS>.
    """
    import pandas as pd

    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = ds["train"].to_pandas()

    df_agg = df.groupby("comment_id").agg(
        text=("text", "first"),
        **{col: (col, "mean") for col in LABEL_COLS},
    ).reset_index()

    for col in LABEL_COLS:
        col_min, col_max = df_agg[col].min(), df_agg[col].max()
        df_agg[col] = (df_agg[col] - col_min) / (col_max - col_min + 1e-9)

    print(f"Dataset shape : {df_agg.shape}")
    return df_agg


def tokenize(texts: list, tokenizer, max_length: int = 128):
    """Tokenise a list of strings. Returns HuggingFace BatchEncoding."""
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors=None)


def make_hf_dataset(encodings, labels) -> Dataset:
    """Build a HuggingFace Dataset from tokeniser output and float label matrix."""
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels.astype(float).tolist(),
    })



class WeightedMSETrainer(Trainer):
    """
    MSE loss with per-label weights inversely proportional to label frequency.
    Pass `label_weights` (torch.Tensor of shape [num_labels]) at construction.
    """

    def __init__(self, *args, label_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = torch.sigmoid(outputs.logits)

        if self.label_weights is not None:
            weights = self.label_weights.to(logits.device)
            loss = (weights * (logits - labels) ** 2).mean()
        else:
            loss = ((logits - labels) ** 2).mean()

        return (loss, outputs) if return_outputs else loss



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = 1 / (1 + np.exp(-logits))  # sigmoid

    preds_bin = (preds > 0.5).astype(int)
    labels_bin = (labels > 0.5).astype(int)

    return {
        "micro_f1": f1_score(labels_bin, preds_bin, average="micro", zero_division=0),
        "macro_f1": f1_score(labels_bin, preds_bin, average="macro", zero_division=0),
        "mse": float(np.mean((preds - labels) ** 2)),
    }



def build_trainer(model, training_args, train_ds, val_ds, label_weights=None):
    return WeightedMSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        label_weights=label_weights,
    )



if __name__ == "__main__":
    df_agg = load_berkeley_data()
    plot_berkeley_label_means(df_agg, LABEL_COLS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_df, val_df = train_test_split(df_agg, test_size=0.1, random_state=42)

    train_enc = tokenize(train_df["text"].tolist(), tokenizer)
    val_enc = tokenize(val_df["text"].tolist(), tokenizer)

    train_ds = make_hf_dataset(train_enc, train_df[LABEL_COLS].values)
    val_ds = make_hf_dataset(val_enc, val_df[LABEL_COLS].values)


    train_labels = train_df[LABEL_COLS].values.astype(float)
    label_means = train_labels.mean(axis=0)
    label_weights = torch.tensor(1.0 / (label_means + 1e-6), dtype=torch.float)
    label_weights = label_weights / label_weights.mean()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_COLS), problem_type="regression"
    )

    training_args = TrainingArguments(
        output_dir="./uc_berkeley_model",
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = build_trainer(model, training_args, train_ds, val_ds, label_weights)
    trainer.train()
    trainer.save_model(DEFAULT_SAVE_PATH)
    tokenizer.save_pretrained(DEFAULT_SAVE_PATH)


    with open(f"{DEFAULT_SAVE_PATH}/labels.json", "w") as f:
        json.dump(LABEL_COLS, f)

    print(f"Model saved to {DEFAULT_SAVE_PATH}")