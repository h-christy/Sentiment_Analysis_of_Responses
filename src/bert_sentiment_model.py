
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data_loader import load_stanford_sentiment



SENTIMENT_MODEL_NAME = "roberta-base"
DEFAULT_SAVE_PATH = "./sentiment_roberta_model_final"


def tokenize_sentiment_data(texts, tokenizer, max_length: int = 128):
    """Tokenise a list/Series of texts. Returns HuggingFace BatchEncoding."""
    return tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
    )


def make_hf_dataset(encodings, labels) -> Dataset:
    """
    Build a HuggingFace Dataset from tokeniser output + float labels.
    Labels should be 1-D (N,) or 2-D (N, 1).
    """
    label_list = (
        labels.astype(np.float32).reshape(-1, 1).tolist()
        if hasattr(labels, "astype")
        else [[float(v)] for v in labels]
    )
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": label_list,
    })



class RegressionTrainer(Trainer):
    """MSE loss with sigmoid activation to keep predictions in [0, 1]."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        loss = nn.MSELoss()(predictions, labels)
        return (loss, outputs) if return_outputs else loss



def compute_metrics_sentiment(eval_pred):
    logits, labels = eval_pred
    predictions = 1 / (1 + np.exp(-logits))  # sigmoid
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {"mse": mse, "r2": r2}



def build_regression_trainer(
    model,
    training_args: TrainingArguments,
    train_ds: Dataset,
    eval_ds: Dataset,
) -> RegressionTrainer:
    return RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_sentiment,
    )


if __name__ == "__main__":
    df = load_stanford_sentiment()
    X = df["sentence"]
    y = df["sentiment_values"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

    train_enc = tokenize_sentiment_data(X_train, tokenizer)
    test_enc = tokenize_sentiment_data(X_test, tokenizer)

    train_ds = make_hf_dataset(train_enc, y_train.values)
    test_ds = make_hf_dataset(test_enc, y_test.values)

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_NAME, num_labels=1, problem_type="regression"
    )

    training_args = TrainingArguments(
        output_dir="./sentiment_roberta_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        eval_strategy="epoch",
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = build_regression_trainer(model, training_args, train_ds, test_ds)
    trainer.train()

    results = trainer.evaluate()
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    trainer.save_model(DEFAULT_SAVE_PATH)
    tokenizer.save_pretrained(DEFAULT_SAVE_PATH)
    print(f"Model saved to {DEFAULT_SAVE_PATH}")