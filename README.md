# End-to-End SMS Spam Classification using Hybrid SVM

## Project Overview
This project implements and evaluates an end-to-end machine learning pipeline for detecting spam SMS messages. The system evolves from a simple text-only baseline to an advanced hybrid architecture that combines TF-IDF text embeddings with engineered structural metadata features. Finally, the model is deployed via an interactive Gradio web interface.

## Architecture Summary
The pipeline consists of several key stages:
1. **Data Ingestion & Preprocessing**: Loading `spam.csv`, removing 403 duplicates to prevent data leakage, and applying text normalization (lowercasing, URL/digit removal, stopword removal, and lemmatization).
2. **Exploratory Data Analysis (EDA)**: Identifying class imbalance and visualizing word clouds to discover high-urgency spam keywords (e.g., "free", "win", "claim").
3. **Baseline Model**: TF-IDF Vectorizer (top 5,000 unigrams/bigrams) + LinearSVC.
4. **Advanced Feature Engineering**: Extracting four structural features: `char_count`, `digit_count`, `caps_ratio`, and `has_url`.
5. **Hybrid Pipeline**: A `ColumnTransformer` that applies TF-IDF to text and `StandardScaler` to numerical features before feeding them into the LinearSVC classifier.
6. **Deployment**: A Gradio interface with dynamic thresholding.

## Key Results & Metrics

| Model | Accuracy | Spam Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Baseline (TF-IDF + SVM)** | ~98.4% | 89.3% | - |
| **Enhanced (Hybrid + SVM)** | **~98.7%** | **91.6%** | **~0.949** |

*The addition of structural features significantly improved the model's sensitivity (Recall) to spam messages, reducing false negatives.*

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SMS-spam-detector.git
   cd SMS-spam-detector
