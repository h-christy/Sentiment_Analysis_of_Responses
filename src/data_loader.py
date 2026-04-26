
import pandas as pd
from sklearn.model_selection import train_test_split


def load_stanford_sentiment(
    labels_path: str = "sentiment_labels.txt",
    dict_path: str = "dictionary.txt",
) -> pd.DataFrame:
    sentiment_numbers = pd.read_csv(labels_path, sep="|")
    sentences = pd.read_csv(dict_path, sep="|", header=None)

    sentences.columns = ["sentence", "phrase_id"]
    sentiment_numbers.columns = ["phrase_id", "sentiment_values"]

    df = pd.merge(sentences, sentiment_numbers, on="phrase_id")
    df = df[["phrase_id", "sentence", "sentiment_values"]]

    df["sentiment_labels"] = pd.cut(
        df["sentiment_values"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["very negative", "negative", "neutral", "positive", "very positive"],
    )
    df = df.dropna()
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "sentiment_labels",
    test_size: float = 0.2,
    random_state: int = 42,
):

    X = df["sentence"]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



if __name__ == "__main__":
    df = load_stanford_sentiment()
    print(df.head())
    print(f"\nTotal rows : {len(df)}")
    print(df["sentiment_labels"].value_counts())