from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd

# Baseline model using Logistic Regression with TF-IDF features
def train_baseline(X_train, y_train, evaluate = True):
    
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train a Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_vectorized, y_train)
    
    # Evaluate the model on the test set
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = log_reg.predict(X_test_vectorized)
    
    if evaluate:
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    
    df = pd.read_csv("data/stanford_sentiment_data.csv")
    X = df['sentence']
    y = df['sentiment_labels']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_baseline(X_train, y_train)