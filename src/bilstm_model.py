
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from data_loader import load_stanford_sentiment, split_data
from visualize import plot_actual_vs_predicted

def build_tokenizer_and_pad(X_train, X_test):

    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    max_sequence_length = max(len(s) for s in X_train_seq)
    vocabulary_size = len(tokenizer.word_index) + 1

    X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding="post")
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding="post")

    print(f"Vocabulary size      : {vocabulary_size}")
    print(f"Max sequence length  : {max_sequence_length}")
    print(f"X_train_padded shape : {X_train_padded.shape}")
    print(f"X_test_padded shape  : {X_test_padded.shape}")

    return X_train_padded, X_test_padded, vocabulary_size, max_sequence_length, tokenizer


def encode_labels(y_train, y_test):

    label_map = {label: i for i, label in enumerate(np.unique(y_train))}
    y_train_enc = np.array([label_map[l] for l in y_train])
    y_test_enc = np.array([label_map[l] for l in y_test])
    num_classes = len(label_map)

    y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)

    print(f"Number of classes    : {num_classes}")
    return y_train_cat, y_test_cat, num_classes, label_map



def build_bilstm_classifier(
    vocabulary_size: int,
    embedding_dim: int,
    max_seq_len: int,
    num_classes: int,
) -> Sequential:

    model = Sequential([
        Embedding(vocabulary_size, embedding_dim, input_length=max_seq_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model


def build_bilstm_regressor(
    vocabulary_size: int,
    embedding_dim: int,
    max_seq_len: int,
) -> Sequential:

    model = Sequential([
        Embedding(vocabulary_size, embedding_dim, input_length=max_seq_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=[MeanAbsoluteError()]
    )
    model.summary()
    return model



def train_model(model, X_train_padded, y_train_data, epochs=5, batch_size=32):
    """Fit model with a 10 % validation split and return the History object."""
    history = model.fit(
        X_train_padded,
        y_train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    return history


def evaluate_regressor(model, X_test_padded, y_test_reg):
    """Evaluate a regression model and print MSE, MAE, R²."""
    loss, mae = model.evaluate(X_test_padded, y_test_reg, verbose=1)
    y_pred = model.predict(X_test_padded)
    r2 = r2_score(y_test_reg, y_pred)

    print(f"BiLSTM Regression MSE : {loss:.4f}")
    print(f"BiLSTM Regression MAE : {mae:.4f}")
    print(f"BiLSTM Regression R²  : {r2:.4f}")

    plot_actual_vs_predicted(y_test_reg, y_pred, title="BiLSTM Regression")
    return loss, mae, r2


if __name__ == "__main__":
    EMBEDDING_DIM = 100
    EPOCHS = 5
    BATCH_SIZE = 32

    df = load_stanford_sentiment()


    X_train, X_test, y_train_cls, y_test_cls = split_data(
        df, target_col="sentiment_labels"
    )
    _, _, y_train_reg, y_test_reg = split_data(df, target_col="sentiment_values")

    X_train_padded, X_test_padded, vocab_size, max_len, _ = build_tokenizer_and_pad(
        X_train, X_test
    )
    y_train_cat, y_test_cat, num_classes, _ = encode_labels(y_train_cls, y_test_cls)


    clf = build_bilstm_classifier(vocab_size, EMBEDDING_DIM, max_len, num_classes)
    train_model(clf, X_train_padded, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE)
    clf.save("bilstm_sentiment_model.h5")


    reg = build_bilstm_regressor(vocab_size, EMBEDDING_DIM, max_len)
    train_model(reg, X_train_padded, y_train_reg, epochs=EPOCHS, batch_size=BATCH_SIZE)
    evaluate_regressor(reg, X_test_padded, y_test_reg)