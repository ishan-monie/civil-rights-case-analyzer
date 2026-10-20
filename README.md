# NLP Final Project — Civil Rights Case Analyzer
### Northwestern University | Text Analytics | Spring 2025

---

## Project Structure

```
project/
├── app/                              ← Interactive tool (YOUR PART)
│   ├── app.py                        ← Main Gradio app — run this
│   └── requirements.txt
│
├── nlp-final-ca-dev/                 ← EDA, cleaning & summarization
│   └── src/
│       ├── preprocessing.ipynb       ← Produces case_data.pkl
│       └── summarize.ipynb           ← T5 summarization pipeline
│
├── nlp-final-class-action-model/     ← Class action prediction models
│   ├── class_action_modeling.ipynb   ← Produces class_action_models.pkl
│   └── src/
│
└── nlp-final-case-type/              ← Case type prediction models
    └── src/
        ├── preprocessing.ipynb
        ├── summarize.ipynb
        └── classification_model.ipynb  ← Produces models/best_model_pipeline.pkl
```

---

## Setup Instructions

### Step 1 — Run teammate notebooks to produce model files

Each notebook must be run **once** (on Colab with GPU recommended) to save the model artifacts.

| Notebook | Output file needed by app |
|---|---|
| `nlp-final-ca-dev/src/preprocessing.ipynb` | `case_data.pkl` (shared input for other notebooks) |
| `nlp-final-class-action-model/class_action_modeling.ipynb` | `nlp-final-class-action-model/class_action_models.pkl` |
| `nlp-final-case-type/src/classification_model.ipynb` | `nlp-final-case-type/models/best_model_pipeline.pkl` |

> Make sure the pkl files are saved in the paths shown above (relative to the project root).

### Step 2 — Install app dependencies

```bash
pip install -r app/requirements.txt
```

### Step 3 — Run the app

From the **project root** directory:

```bash
python app/app.py
```

This will print a local URL (e.g. `http://127.0.0.1:7860`) and a public shareable link.

### Running in Google Colab

```python
%cd /content/project
!pip install -r app/requirements.txt -q
!python app/app.py
```

The `share=True` flag in `app.py` will generate a public `gradio.live` link valid for 72 hours.

---

## What the App Does

| Input | A raw legal case (real or fictional) |
|---|---|
| **Long summary** | Multi-paragraph T5 summary |
| **Short summary** | Single-paragraph T5 summary |
| **Tiny summary** | One-sentence T5 summary |
| **Class action prediction** | Yes / No — TF-IDF + Logistic Regression (96.4% accuracy) |
| **Case type prediction** | Employment Discrimination / Immigration & Border / Confinement Conditions / Other Civil Rights |

---

## Model Details

### Summarization
- Model: `t5-small` (HuggingFace)
- Strategy: Two-stage chunked summarization (input → chunk → intermediate summaries → final pass)
- Input: `cleaned_text` column from preprocessing

### Class Action (binary)
- Best model: Logistic Regression + TF-IDF (30k features, unigrams+bigrams)
- Test accuracy: 96.4% | Macro F1: 95.8% | ROC-AUC: 0.988
- Artifacts: `label_encoder`, `tfidf_vectorizer`, `lr_tfidf_model` inside `class_action_models.pkl`

### Case Type (4-class)
- Best model: LinearSVC + TF-IDF (10k features, unigrams+bigrams)
- Feature source: `summary_long` column
- Artifacts: `vectorizer` + `model` inside `models/best_model_pipeline.pkl`
- Groups: Employment Discrimination · Immigration & Border · Confinement Conditions · Other Civil Rights
