import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_label_distribution(
    df: pd.DataFrame,
    label_col: str = "sentiment_labels",
    title: str = "Label Distribution",
) -> None:
    """Bar chart of value counts for a categorical label column."""
    counts = df[label_col].value_counts()
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.astype(str), counts.values, color="skyblue")
    plt.title(title)
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    title: str = "Actual vs Predicted Values",
) -> None:
    """Scatter plot of true vs predicted continuous values with an ideal line."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Sentiment")
    plt.ylabel("Predicted Sentiment")
    plt.title(title)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red",
        linestyle="--",
    )
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    labels: list,
    values: list,
    title: str = "Model Comparison",
    ylabel: str = "Score",
) -> None:
    """Line plot comparing metric values across models."""
    plt.figure(figsize=(6, 4))
    plt.plot(labels, values, marker="o")
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_berkeley_label_means(df: pd.DataFrame, label_cols: list) -> None:
    """Bar chart of mean values for each Berkeley hate-speech label."""
    means = [df[col].mean() for col in label_cols]
    plt.figure(figsize=(10, 5))
    plt.bar(label_cols, means, color="skyblue")
    plt.title("UC Berkeley Data — Label Means")
    plt.xlabel("Labels")
    plt.xticks(rotation=45)
    plt.ylabel("Mean")
    plt.tight_layout()
    plt.show()