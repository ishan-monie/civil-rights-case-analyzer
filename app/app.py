"""
Civil Rights Case Analyzer — Interactive Tool
==============================================
Loads models saved by teammates and provides a Gradio UI that:
  1. Generates long / short / tiny summaries (T5, from nlp-final-ca-dev)
  2. Predicts class_action_sought  (TF-IDF + LogReg, from nlp-final-class-action-model)
  3. Predicts case_type group      (TF-IDF + LinearSVC, from nlp-final-case-type)

Run from the PROJECT ROOT:
    pip install -r app/requirements.txt
    python app/app.py

Model files expected (run each teammate's notebook first to produce them):
    nlp-final-class-action-model/class_action_models.pkl
    nlp-final-case-type/models/best_model_pipeline.pkl
"""

import re
import string
import pickle
import warnings
import numpy as np
import gradio as gr

warnings.filterwarnings("ignore")

# ── NLTK setup ────────────────────────────────────────────────────────────────
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Transformers (T5 summarizer) ──────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

# =============================================================================
# 1. LOAD MODELS
# =============================================================================

print("Loading models…")

# ── Summarizer (T5-small, same as nlp-final-ca-dev/src/summarize.ipynb) ───────
t5_pipe = hf_pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small",
    framework="pt",
    device=-1,          # CPU; set device=0 for GPU
)

# ── Class-action model (from nlp-final-class-action-model) ───────────────────
CA_MODEL_PATH = "nlp-final-class-action-model/class_action_models.pkl"
with open(CA_MODEL_PATH, "rb") as f:
    ca_artifacts = pickle.load(f)

ca_le        = ca_artifacts["label_encoder"]
ca_tfidf     = ca_artifacts["tfidf_vectorizer"]
ca_lr        = ca_artifacts["lr_tfidf_model"]   # best model per teammate's summary

# ── Case-type model (from nlp-final-case-type) ────────────────────────────────
CT_MODEL_PATH = "nlp-final-case-type/models/best_model_pipeline.pkl"
with open(CT_MODEL_PATH, "rb") as f:
    ct_pipeline = pickle.load(f)

ct_vectorizer = ct_pipeline["vectorizer"]
ct_model      = ct_pipeline["model"]           # LinearSVC

# Case-type label map (matches teammate's notebook)
CT_LABEL_MAP = {
    "Employment Discrimination": "⚖️ Employment Discrimination",
    "Immigration & Border":      "🛂 Immigration & Border",
    "Confinement Conditions":    "🏛️ Confinement Conditions",
    "Other Civil Rights":        "📋 Other Civil Rights",
}

print("All models loaded.")

# =============================================================================
# 2. TEXT PREPROCESSING  (mirrors preprocessing.ipynb exactly)
# =============================================================================

_stop_words  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()
_re_url      = re.compile(r"https?://\S+|www\.\S+")
_re_cite     = re.compile(r"\b\d+\s+(?:U\.S\.|F\.3d|F\.2d|F\.Supp\.?\d*|S\.Ct\.|L\.Ed\.2d)\s+\d+\b")
_re_fn       = re.compile(r"\[\d+\]")
_re_docket   = re.compile(r"Case\s+\d+:\d+-[a-z]+-\d+[^\n]{0,120}")
_re_space    = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light cleaning — mirrors preprocessing.ipynb clean_text()."""
    if not text:
        return ""
    text = text.replace("\x0c", " ")
    text = _re_url.sub("", text)
    text = _re_cite.sub("", text)
    text = _re_fn.sub("", text)
    text = _re_docket.sub("", text)
    return _re_space.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Full normalization — mirrors preprocessing.ipynb normalize_text()."""
    text = clean_text(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)


# =============================================================================
# 3. SUMMARIZATION  (mirrors summarize.ipynb)
# =============================================================================

def _chunk_text(text: str, chunk_size: int = 1200):
    words = text.split()
    chunks, current, current_len = [], [], 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def _t5_summarize_chunk(text: str, max_length: int = 80, min_length: int = 15) -> str:
    result = t5_pipe(
        f"summarize: {text}",
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]


LEVEL_PARAMS = {
    "long":  dict(max_length=400, min_length=100),
    "short": dict(max_length=150, min_length=40),
    "tiny":  dict(max_length=50,  min_length=10),
}


def generate_summary(text: str, level: str, max_input_chars: int = 8000, max_chunks: int = 5) -> str:
    """Two-stage T5 summarization — mirrors summarize.ipynb generate_t5_summary()."""
    cleaned = clean_text(text)
    truncated = cleaned[:max_input_chars]
    chunks = _chunk_text(truncated, chunk_size=1200)[:max_chunks]
    if not chunks:
        return "Not enough text to summarize."
    intermediates = [_t5_summarize_chunk(c, max_length=80, min_length=15) for c in chunks]
    combined = " ".join(intermediates)
    return _t5_summarize_chunk(combined, **LEVEL_PARAMS[level])


# =============================================================================
# 4. PREDICTIONS
# =============================================================================

def predict_class_action(raw_text: str) -> tuple[str, float]:
    """Returns (label, confidence_pct) using TF-IDF + LogReg."""
    normed = normalize_text(raw_text)
    vec    = ca_tfidf.transform([normed])
    probs  = ca_lr.predict_proba(vec)[0]
    idx    = int(np.argmax(probs))
    label  = ca_le.inverse_transform([idx])[0]
    conf   = round(float(probs[idx]) * 100, 1)
    return label, conf


def predict_case_type(raw_text: str) -> tuple[str, str]:
    """Returns (raw_label, display_label) using TF-IDF + LinearSVC.
    LinearSVC has no predict_proba, so we use decision function distance instead.
    """
    cleaned = clean_text(raw_text)
    vec     = ct_vectorizer.transform([cleaned])
    label   = ct_model.predict(vec)[0]
    display = CT_LABEL_MAP.get(label, label)
    # Decision function gives distance to boundary — larger = more confident
    scores  = ct_model.decision_function(vec)[0]
    confidence_note = "(high confidence)" if abs(scores.max()) > 1.0 else "(moderate confidence)"
    return display, confidence_note


# =============================================================================
# 5. MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_case(case_text: str):
    if not case_text or len(case_text.strip().split()) < 30:
        msg = "⚠️ Please enter at least 30 words of case text."
        return msg, "", "", "", ""

    # Summaries
    long_sum  = generate_summary(case_text, "long")
    short_sum = generate_summary(case_text, "short")
    tiny_sum  = generate_summary(case_text, "tiny")

    # Predictions
    ca_label, ca_conf         = predict_class_action(case_text)
    ct_display, ct_conf_note  = predict_case_type(case_text)

    ca_out = f"{ca_label}  —  {ca_conf}% confidence"
    ct_out = f"{ct_display}  {ct_conf_note}"

    return long_sum, short_sum, tiny_sum, ca_out, ct_out


# =============================================================================
# 6. GRADIO UI
# =============================================================================

EXAMPLES = [
    [
        "Plaintiffs, a class of current and former prisoners at Cook County Jail, filed suit "
        "against the Sheriff alleging unconstitutional conditions of confinement including "
        "inadequate medical care, excessive use of force by guards, and failure to protect "
        "inmates from violence by other prisoners. They seek class certification on behalf of "
        "all similarly situated detainees and request injunctive relief to remedy systemic "
        "constitutional violations. The complaint alleges deliberate indifference under the "
        "Eighth and Fourteenth Amendments."
    ],
    [
        "The EEOC filed suit on behalf of a group of female employees at a regional bank who "
        "were systematically passed over for promotion in favor of less-qualified male "
        "colleagues. The complaint alleges violations of Title VII, including discriminatory "
        "pay practices, a hostile work environment, and retaliation against those who filed "
        "internal complaints. The agency seeks monetary relief and a consent decree requiring "
        "the bank to implement non-discriminatory promotion policies."
    ],
    [
        "Petitioners, a group of asylum seekers detained at the southern border, challenge the "
        "government's policy of mandatory detention pending removal proceedings. They allege "
        "violations of the Immigration and Nationality Act and the Due Process Clause of the "
        "Fifth Amendment, arguing that prolonged detention without individualized hearings is "
        "unconstitutional. Petitioners seek a writ of habeas corpus and immediate bond hearings."
    ],
]

with gr.Blocks(title="Civil Rights Case Analyzer", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # ⚖️ Civil Rights Case Analyzer
        ### Northwestern University — Text Analytics Final Project (Spring 2025)
        Paste a real or fictional civil-rights legal case to receive:
        **three auto-generated summaries** and **two model predictions**.
        """
    )

    with gr.Row():
        case_input = gr.Textbox(
            label="Case Text",
            placeholder="Paste the full legal case text here (minimum ~30 words)…",
            lines=14,
            scale=1,
        )

    analyze_btn = gr.Button("🔍 Analyze Case", variant="primary", size="lg")

    # ── Summaries ──────────────────────────────────────────────────────────────
    gr.Markdown("---\n## 📝 Generated Summaries")
    gr.Markdown(
        "_Summaries are generated by a two-stage T5-small pipeline "
        "(chunked → intermediate → final pass)._"
    )

    long_out  = gr.Textbox(label="Long Summary  (multi-paragraph)", lines=8)

    with gr.Row():
        short_out = gr.Textbox(label="Short Summary  (one paragraph)", lines=4, scale=2)
        tiny_out  = gr.Textbox(label="Tiny Summary  (one sentence)",   lines=2, scale=1)

    # ── Predictions ────────────────────────────────────────────────────────────
    gr.Markdown("---\n## 🔍 Model Predictions")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Class Action Sought?")
            gr.Markdown(
                "_TF-IDF + Logistic Regression · 96.4% test accuracy · 0.988 ROC-AUC_"
            )
            ca_out = gr.Textbox(label="Prediction", lines=1)

        with gr.Column():
            gr.Markdown("### Case Type Group")
            gr.Markdown(
                "_TF-IDF + Legal-BERT LinearSVC · 4 grouped categories_"
            )
            ct_out = gr.Textbox(label="Prediction", lines=1)

    # ── Examples ───────────────────────────────────────────────────────────────
    gr.Markdown("---\n### 💡 Try an Example")
    gr.Examples(
        examples=EXAMPLES,
        inputs=case_input,
        label="Click any example to load it, then press Analyze Case",
    )

    # ── Wire up ────────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_case,
        inputs=case_input,
        outputs=[long_out, short_out, tiny_out, ca_out, ct_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)   # share=True generates a public link (works in Colab too)
