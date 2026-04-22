import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_dataframe():
    sentiment_numbers = pd.read_csv('data/stanford/sentiment_labels.txt', sep='|')
    sentences = pd.read_csv('data/stanford/dictionary.txt', sep='|',header=None)
    sentences.columns = ['sentence','phrase_id' ]
    sentiment_numbers.columns = ['phrase_id','sentiment_values']

    df = pd.merge(sentences, sentiment_numbers, on='phrase_id')
    df = df[['phrase_id','sentence', 'sentiment_values']]
    df['sentiment_labels'] = pd.cut(df['sentiment_values'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['very negative', 'negative', 'neutral', 'positive', 'very positive'])

    df = df.dropna()
    return df

def bar_chart(df):

    labels = []
    label_values = []
    df['sentiment_labels'] = df['sentiment_labels'].fillna('negative')

    for label in df['sentiment_labels'].unique():
        labels.append(label)
        label_values.append(len(df[df['sentiment_labels'] == label]))


    plt.bar(labels, label_values, color='skyblue')
    plt.title('Basic Bar Plot')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.show()

if __name__ == "__main__":
    df = create_dataframe()
    df.to_csv('data/stanford_sentiment_data.csv', index=False)
    bar_chart(df)