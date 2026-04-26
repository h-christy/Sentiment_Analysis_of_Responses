# NLP Sentiment & Toxicity Pipeline

---

## File Structure

```
nlp_project/
├── data_loader.py            # Stanford Sentiment data loading & splitting
├── visualize.py              # Shared plotting utilities
├── baseline_model.py         # TF-IDF + Logistic/Linear Regression baselines
├── bilstm_model.py           # BiLSTM classifier + regressor (TensorFlow/Keras)
├── bert_sentiment_model.py   # RoBERTa-base regression model (HuggingFace)
├── berkeley_model.py         # Twitter-RoBERTa multi-label toxicity model
├── twitter_inference.py      # Apply models to Twitter CSV data
└── chatgpt_inference.py      # Apply models to lmsys/lmsys-chat-1m dataset
```

---

## Dependency Graph

```
data_loader  ──────────────────────────────────┐
     │                                          │
     ├──► baseline_model ──► visualize          │
     │                                          │
     ├──► bilstm_model   ──► visualize          │
     │                                          │
     └──► bert_sentiment_model                  │
                                                │
berkeley_model ──► visualize                    │
                                                │
twitter_inference ──► bert_sentiment_model      │
                  ──► berkeley_model            │
                                                │
chatgpt_inference ──► bert_sentiment_model      │
                  ──► berkeley_model            │
```

---

## Running Each Stage

### 1 — Baseline (TF-IDF)
```bash
python baseline_model.py
```

### 2 — BiLSTM
```bash
python bilstm_model.py
```

### 3 — Train RoBERTa Sentiment Model
```bash
python bert_sentiment_model.py
```

### 4 — Train Berkeley Toxicity Model
```bash
python berkeley_model.py
```

### 5 — Twitter Inference
```bash
python twitter_inference.py \
    --input tweets.csv \
    --output tweets_with_scores.csv \
    --sentiment_model_path ./sentiment_roberta_model_final \
    --berkeley_model_path  ./uc_berkeley_model_best
```

### 6 — ChatGPT Inference
```bash
python chatgpt_inference.py \
    --output chatgpt_anonymous_df.csv \
    --sentiment_model_path ./sentiment_roberta_model_final \
    --berkeley_model_path  ./uc_berkeley_model_best \
    --sample 200000
```

---

## Required Data Files

| File | Used by |
|------|---------|
| `sentiment_labels.txt` | `data_loader.py` |
| `dictionary.txt`       | `data_loader.py` |
| `tweets.csv`           | `twitter_inference.py` |

Berkeley and ChatGPT datasets are downloaded automatically from HuggingFace.

---

## Key Dependencies

```
pandas scikit-learn tensorflow transformers datasets torch langdetect matplotlib seaborn
```
