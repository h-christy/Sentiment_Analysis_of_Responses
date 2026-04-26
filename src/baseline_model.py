from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
import pandas as pd

from data_loader import load_stanford_sentiment, split_data
from visualize import plot_actual_vs_predicted


def train_classification_baseline(X_train, X_test, y_train, y_test, evaluate: bool = True):

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_vec, y_train)

    if evaluate:
        y_pred = log_reg.predict(X_test_vec)
        print("=== Classification Baseline ===")
        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))
        print("Classification Report:\n", classification_report(y_test, y_pred))

    return vectorizer, log_reg



def train_regression_baseline(X_train, X_test, y_train, y_test):

    vectorizer_reg = TfidfVectorizer()
    X_train_vec = vectorizer_reg.fit_transform(X_train)
    X_test_vec = vectorizer_reg.transform(X_test)

    linear_reg = LinearRegression()
    linear_reg.fit(X_train_vec, y_train)

    y_pred = linear_reg.predict(X_test_vec)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== Regression Baseline ===")
    print(f"MSE : {mse:.4f}")
    print(f"R²  : {r2:.4f}")

    plot_actual_vs_predicted(y_test, y_pred, title="Baseline Linear Regression")

    return vectorizer_reg, linear_reg



if __name__ == "__main__":
    df = load_stanford_sentiment()

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_data(
        df, target_col="sentiment_labels"
    )
    train_classification_baseline(X_train_cls, X_test_cls, y_train_cls, y_test_cls)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
        df, target_col="sentiment_values"
    )
    train_regression_baseline(X_train_reg, X_test_reg, y_train_reg, y_test_reg)