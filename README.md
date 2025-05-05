
# Weaponization of Migration Flows (WOM) Classification

Ever wondered how transit states turn migrant flows into a diplomatic bargaining chip? In this project, we develop and compare two NLP pipelines to automatically detect when political leaders threaten to “open the borders” or criticize destination countries’ migration policies—a tactic we term the **Weaponization of Migration Flows (WOM)**.

## Project Overview

I focus on all English‐language speeches by Turkish President Erdoğan (2014–2025), web‐scraped from the official presidency website. After manually annotating a balanced sample, I train:

* **Logistic Regression + TF–IDF**
  (with spaCy/NLTK preprocessing & SMOTE)
* **Fine-tuned DistilBERT**
  (with Hugging Face Transformers)

and evaluate both on a held-out test set as well as via 5-fold cross-validation.

## Goals

* **Quantify WOM** by detecting “threat” or “criticism” rhetoric in presidential speeches
* **Compare classical vs. transformer-based models**
* **Demonstrate how contextual embeddings** improve detection recall (critical for catching true threats)
* **Provide a reusable pipeline** for analyzing WOM in other transit-migration contexts

## Methods

### Data Acquisition

1. Web-scraped 242 speeches → 3,976 paragraphs
2. Filtered out “visit”/“phone call” notices & salutations

### Annotation

1. Hand-label 472 random paragraphs → only 16 positive WOM
2. Zero-shot classification (`facebook/bart-large-mnli`) → hand-verify → 338 WOM-positive + 500 WOM-negative

### Baseline Model

* **Preprocessing:** spaCy tokenization & lemmatization + NLTK stop-word removal + custom NER tokens
* **Features:** `TfidfVectorizer` (grid-searched n-grams & `min_df`)
* **Balancing:** SMOTE
* **Classifier:** `LogisticRegression`

### Advanced Model

* **Preprocessing:** spaCy/NLTK pipeline → BERT tokenization
* **Model:** Fine-tune `distilbert-base-uncased` for 3 epochs
* **Library:** Hugging Face Transformers

### Evaluation

* **Hold-out (20% data):** Accuracy, Precision, Recall, F₁
* **5-Fold CV:** mean ± std accuracy

## Key Results

| Model                           | Accuracy | Precision | Recall |   F₁   |
| :------------------------------ | :------: | :-------: | :----: | :----: |
| Logistic Regression with TF–IDF |  0.7143  |   0.7143  | 0.7224 | 0.7117 |
| Advanced DistilBERT Model       |  0.8095  |   0.7195  | 0.8676 | 0.7867 |

* **Recall Boost:** +14.5 pts (72.2 → 86.8) reduces missed threats
* **F₁ Improvement:** +7.5 pts (71.2 → 78.7) for balanced detection
* **5-Fold CV (DistilBERT):** 84.3 % ± 2.9 % accuracy (stable performance)

## Repository Structure

```
sp2025-migration/
├── notebooks/
│   ├── wom_data_webscraping.ipynb
│   ├── wom_zeroshot_classification.ipynb
│   └── wom_nlp_model.ipynb
├── models/
│   └── wom_nlp_model.ipynb
├── README.Rmd            
└── report/
    └── report.pdf        
```

## Further Reading

For a detailed walkthrough, see the full project report:

Alan, B. (2025). *An NLP Approach to Weaponization of Migration Flows*. Tulane University.
\[report/Report.pdf]

---

*Developed by Baris Alan, Tulane University ([balan@tulane.edu](mailto:balan@tulane.edu))*

```
```
