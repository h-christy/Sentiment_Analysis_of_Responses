import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def tokenize_and_pad(X_train, X_test, y_train, y_test):
    tokenizer = Tokenizer(num_words=20000, oov_token="<unk>") #
    tokenizer.fit_on_texts(X_train)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    vocabulary_size = len(tokenizer.word_index) + 1
    max_sequence_length = max([len(x) for x in X_train_sequences]) # Get max length from training data

    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post')

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Max sequence length: {max_sequence_length}")
    print(f"Shape of X_train_padded: {X_train_padded.shape}")
    print(f"Shape of X_test_padded: {X_test_padded.shape}")
    
    label_map = {label: i for i, label in enumerate(np.unique(y_train))}
    y_train_encoded = np.array([label_map[label] for label in y_train])
    y_test_encoded = np.array([label_map[label] for label in y_test])

    num_classes = len(label_map)
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)

    print(f"Number of classes: {num_classes}")
    print(f"Shape of y_train_categorical: {y_train_categorical.shape}")
    print(f"Shape of y_test_categorical: {y_test_categorical.shape}")

    return X_train_padded, X_test_padded, y_train_categorical, y_test_categorical, vocabulary_size

def build_bilstm_model(vocabulary_size, max_sequence_length, num_classes, X_train_padded, y_train_categorical, X_test_padded, y_test_categorical):
    embedding_dim = 100 # You can adjust this value

    model = Sequential([
        Embedding(vocabulary_size, embedding_dim, input_length=max_sequence_length),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    epochs = 5 
    batch_size = 32 

    history = model.fit(
    X_train_padded, y_train_categorical,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1, 
    verbose=1
    )
    
    model.save('models/bilstm_sentiment_model.h5')
    
    model.evaluate(X_test_padded, y_test_categorical, verbose=1)
    
    return model, history
if __name__ == "__main__":
    df = pd.read_csv('data/stanford_sentiment_treebank.csv')
    X = df['sentence']
    y = df['sentiment_labels']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_padded, X_test_padded, y_train_categorical, y_test_categorical, vocabulary_size = tokenize_and_pad(X_train, X_test, y_train, y_test)
    
    build_bilstm_model(vocabulary_size, X_train_padded.shape[1], y_train_categorical.shape[1], X_train_padded, y_train_categorical, X_test_padded, y_test_categorical)