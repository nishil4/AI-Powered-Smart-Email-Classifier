import json
import os
import re
import csv
import threading
import warnings
from datetime import datetime
from typing import List, Optional
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE2_MODEL_PATH = os.path.join(BASE_DIR, "module2_trained_models", "best_email_classifier.pkl")
MODULE3_MODEL_PATH = os.path.join(BASE_DIR, "module3_trained_models", "best_urgency_classifier.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "module3_trained_models", "tfidf_vectorizer.pkl")
RULE_CONFIG_PATH = os.path.join(BASE_DIR, "module3_trained_models", "urgency_rule_config.json")
DEFAULT_STORE_DIR = "/var/data" if os.path.isdir("/var/data") else os.path.join(BASE_DIR, "dashboard_data")
STORE_DIR = os.getenv("STORE_DIR", DEFAULT_STORE_DIR)
STORE_PATH = os.path.join(STORE_DIR, "email_predictions_log.csv")
OUTBOX_DIR = os.path.join(BASE_DIR, "integration_outbox")
STORE_COLUMNS = ["timestamp", "source", "subject", "email_text", "predicted_category", "predicted_urgency", "technical_category"]
STORE_LOCK = threading.Lock()

CATEGORY_MAP = {
    "forum": "forum",
    "updates": "updates",
    "verify_code": "verify_code",
    "social_media": "social_media",
    "promotions": "promotions",
    "spam": "spam",
}

CATEGORY_FEATURES = [
    "category_forum",
    "category_promotions",
    "category_social_media",
    "category_spam",
    "category_updates",
    "category_verify_code",
]

RULE_SIGNAL_COLS = [
    "high_count",
    "medium_count",
    "low_count",
    "verification_score",
    "security_score",
    "phishing_score",
    "exclamations",
    "has_numeric_code",
    "caps_ratio",
    "text_length",
    "spam_default_low",
]

LABEL_TO_IDX = {"low": 0, "medium": 1, "high": 2}
IDX_TO_LABEL = {0: "low", 1: "medium", 2: "high"}


class IngestEmailRequest(BaseModel):
    source: str = Field(default="Gmail", description="Gmail source")
    subject: str = Field(default="", description="Email subject")
    body: str = Field(..., min_length=1, description="Email body")
    attachments: Optional[List[str]] = Field(default_factory=list, description="Attachment names")
    target_systems: Optional[List[str]] = Field(default_factory=list, description="Example: ['crm', 'ticketing']")
    callback_url: Optional[str] = Field(default=None, description="Optional webhook URL for forwarding prediction")


class PredictionResponse(BaseModel):
    timestamp: str
    source: str
    subject: str
    predicted_category: str
    predicted_urgency: str
    technical_category: str
    routed_systems: List[str]
    callback_sent: bool


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"Sent from my.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Best regards.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Thanks.*\n.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Sincerely.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_rule_keywords():
    if os.path.exists(RULE_CONFIG_PATH):
        with open(RULE_CONFIG_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("urgency_keywords", {})
    return {
        "high": [
            "urgent", "asap", "immediately", "critical", "emergency", "outage", "production down",
            "security alert", "unauthorized", "failed login", "account locked", "account suspended",
            "verification code", "otp", "password reset"
        ],
        "medium": [
            "request", "need", "please", "kindly", "follow up", "update", "review", "confirm",
            "action needed", "required", "pending", "deadline", "waiting", "soon", "approval"
        ],
        "low": ["fyi", "for your reference", "no action needed", "newsletter", "digest", "general update"],
    }


def category_rule_fallback(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["verification code", "verify code", "otp", "pin", "passcode", "authentication code", "access code", "account verification", "password reset"]):
        return "verify_code"
    if any(k in t for k in ["click here", "limited time", "winner", "unsubscribe", "scam", "fake", "claim prize"]):
        return "spam"
    if any(k in t for k in ["offer", "free", "discount", "sale", "deal", "coupon", "clearance", "exclusive"]):
        return "promotions"
    if any(k in t for k in ["like", "comment", "follow", "friend request", "event invitation", "rsvp", "group"]):
        return "social_media"
    if any(k in t for k in ["thread", "wiki", "moderator", "forum", "discussion", "post moved", "reply"]):
        return "forum"
    return "updates"


def urgency_rule_score(text: str, category: str, urgency_keywords: dict) -> dict:
    text_raw = "" if text is None else str(text)
    text_low = text_raw.lower()
    category_low = "" if category is None else str(category).lower()

    high_count = sum(keyword in text_low for keyword in urgency_keywords.get("high", []))
    medium_count = sum(keyword in text_low for keyword in urgency_keywords.get("medium", []))
    low_count = sum(keyword in text_low for keyword in urgency_keywords.get("low", []))

    verification_score = sum(keyword in text_low for keyword in [
        "verification code", "verify code", "otp", "authentication code", "security code",
        "pin code", "passcode", "password reset", "two-factor", "2fa", "account verification",
    ])
    security_score = sum(keyword in text_low for keyword in [
        "security alert", "unauthorized", "failed login", "account locked", "account suspended",
        "suspicious activity", "compromised", "blocked", "hacked", "breach",
    ])
    phishing_score = sum(keyword in text_low for keyword in [
        "click here", "act now", "bit.ly", "fake", "scam", "limited time",
    ])

    problem_keywords = ["error", "failed", "broken", "not working", "crash", "down", "outage",
                       "issue", "problem", "urgent fix", "critical bug"]
    problem_score = sum(keyword in text_low for keyword in problem_keywords)

    exclamations = text_low.count("!")
    question_marks = text_low.count("?")
    has_numeric_code = int(bool(re.search(r"\b\d{4,8}\b", text_low)))
    caps_ratio = sum(1 for ch in text_raw if ch.isupper()) / max(len(text_raw), 1)
    text_length = len(text_low.split())
    word_count = len(text_raw.split())
    sentence_count = len(re.split(r'[.!?]', text_low)) - 1

    spam_default_low = int(category_low == "spam")

    urgent_phrases = ["as soon as possible", "immediately", "right now", "asap", "urgent", "critical",
                      "must be done", "needs to be fixed", "production issue"]
    urgent_score = sum(phrase in text_low for phrase in urgent_phrases)

    very_short = word_count < 5
    very_long = word_count > 300
    many_questions = question_marks >= 2

    if verification_score >= 1 or security_score >= 1 or urgent_score >= 1 or (high_count >= 2 and problem_score >= 1):
        rule_label = "high"
    elif (problem_score >= 1 and (exclamations >= 2 or urgent_score >= 1)) or (medium_count >= 2) or (phishing_score >= 2) or (exclamations >= 3):
        rule_label = "medium"
    elif low_count >= 2 or (spam_default_low == 1 and high_count == 0 and medium_count == 0):
        rule_label = "low"
    else:
        rule_label = "low"

    return {
        "rule_label": rule_label,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "verification_score": verification_score,
        "security_score": security_score,
        "phishing_score": phishing_score,
        "problem_score": problem_score,
        "urgent_score": urgent_score,
        "exclamations": exclamations,
        "question_marks": question_marks,
        "has_numeric_code": has_numeric_code,
        "caps_ratio": caps_ratio,
        "text_length": text_length,
        "word_count": word_count,
        "sentence_count": max(1, sentence_count),
        "many_questions": int(many_questions),
        "very_short": int(very_short),
        "very_long": int(very_long),
        "spam_default_low": spam_default_low,
    }


def softmax_np(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(logits)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def get_ml_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        decision = model.decision_function(X)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        proba = softmax_np(decision)

    model_classes = list(model.classes_)
    proba_aligned = np.zeros((proba.shape[0], 3), dtype=float)
    for col_idx, cls in enumerate(model_classes):
        if cls in LABEL_TO_IDX:
            proba_aligned[:, LABEL_TO_IDX[cls]] = proba[:, col_idx]
    row_sums = proba_aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba_aligned / row_sums


def rule_prob_vector(signal_row):
    low_score = 0.8 + 0.3 * signal_row["low_count"] + 0.1 * signal_row["spam_default_low"]
    med_score = (
        1.2
        + 1.5 * signal_row["medium_count"]
        + 0.8 * signal_row["phishing_score"]
        + 0.5 * signal_row["problem_score"]
        + 0.3 * signal_row["exclamations"]
        + 0.4 * signal_row["question_marks"]
        + 0.2 * signal_row["many_questions"]
    )
    high_score = (
        1.0
        + 1.6 * signal_row["high_count"]
        + 2.5 * signal_row["verification_score"]
        + 2.5 * signal_row["security_score"]
        + 1.8 * signal_row["urgent_score"]
        + 1.2 * signal_row["problem_score"]
        + 0.5 * signal_row["has_numeric_code"]
        + 0.3 * signal_row["caps_ratio"]
    )
    vec = np.array([low_score, med_score, high_score], dtype=float)
    return vec / vec.sum()


def hybrid_predict_single(ml_prob, rule_prob, signal_row, alpha=0.85, conf_threshold=0.40, strong_signal_threshold=2):
    fused = alpha * ml_prob + (1 - alpha) * rule_prob
    pred_idx = int(np.argmax(fused))
    ml_conf = float(np.max(ml_prob))
    ml_pred = int(np.argmax(ml_prob))

    strong_signal = (
        signal_row["verification_score"] + signal_row["security_score"] + 
        signal_row["urgent_score"] + signal_row["problem_score"]
    ) >= strong_signal_threshold

    if signal_row["verification_score"] >= 1 or signal_row["security_score"] >= 1:
        pred_idx = LABEL_TO_IDX["high"]
    elif signal_row["urgent_score"] >= 1 or (signal_row["problem_score"] >= 1 and signal_row["exclamations"] >= 2):
        pred_idx = LABEL_TO_IDX["high"]
    elif ml_conf < conf_threshold and strong_signal:
        pred_idx = LABEL_TO_IDX["high"]
    elif (signal_row["phishing_score"] >= 1 or signal_row["medium_count"] >= 1 or signal_row["problem_score"] >= 1) and ml_pred == LABEL_TO_IDX["low"]:
        if ml_conf < (conf_threshold + 0.08):
            pred_idx = LABEL_TO_IDX["medium"]
    elif signal_row["many_questions"] == 1 and ml_pred == LABEL_TO_IDX["low"]:
        pred_idx = LABEL_TO_IDX["medium"]

    return IDX_TO_LABEL[pred_idx]


def build_urgency_feature_vector(text: str, technical_category: str, tfidf_vectorizer, urgency_keywords: dict):
    signal_row = urgency_rule_score(text, technical_category, urgency_keywords)
    signal_df = pd.DataFrame([signal_row])

    category_feature_row = {k: 0 for k in CATEGORY_FEATURES}
    cat_key = f"category_{str(technical_category).lower()}"
    if cat_key in category_feature_row:
        category_feature_row[cat_key] = 1

    signal_features = signal_df[RULE_SIGNAL_COLS].astype(float)
    cat_features = pd.DataFrame([category_feature_row]).astype(float)
    tabular = pd.concat([signal_features.reset_index(drop=True), cat_features.reset_index(drop=True)], axis=1)

    tfidf_vec = tfidf_vectorizer.transform([text]).toarray()
    combined = np.hstack([tfidf_vec, tabular.to_numpy(dtype=float)])
    return combined, signal_row


def ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    if not os.path.exists(STORE_PATH):
        pd.DataFrame(columns=STORE_COLUMNS).to_csv(STORE_PATH, index=False)


def append_store(record: dict):
    ensure_store()
    row = {col: record.get(col) for col in STORE_COLUMNS}
    with STORE_LOCK:
        file_exists = os.path.exists(STORE_PATH)
        with open(STORE_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STORE_COLUMNS)
            if not file_exists or os.path.getsize(STORE_PATH) == 0:
                writer.writeheader()
            writer.writerow(row)


def read_store_df() -> pd.DataFrame:
    ensure_store()
    try:
        with STORE_LOCK:
            return pd.read_csv(STORE_PATH)
    except Exception:
        return pd.DataFrame(columns=STORE_COLUMNS)


def route_to_systems(payload: dict, target_systems: List[str], callback_url: Optional[str]):
    routed = []

    for system in target_systems:
        routed.append(system)
        out_path = os.path.join(OUTBOX_DIR, f"{system.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    callback_sent = False
    if callback_url:
        try:
            body = json.dumps(payload).encode("utf-8")
            req = Request(callback_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=5):
                callback_sent = True
        except Exception:
            callback_sent = False

    return routed, callback_sent


app = FastAPI(title="Smart Email Classifier Integration API", version="1.0.0")

category_model = joblib.load(MODULE2_MODEL_PATH) if os.path.exists(MODULE2_MODEL_PATH) else None
urgency_model = joblib.load(MODULE3_MODEL_PATH) if os.path.exists(MODULE3_MODEL_PATH) else None
tfidf_vectorizer = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None
urgency_keywords = load_rule_keywords()


@app.get("/health")
def health():
    current_count = int(len(read_store_df()))
    return {
        "status": "ok",
        "category_model_loaded": category_model is not None,
        "urgency_model_loaded": urgency_model is not None,
        "tfidf_loaded": tfidf_vectorizer is not None,
        "record_count": current_count,
        "store_path": STORE_PATH,
    }


@app.get("/predictions")
def get_predictions(
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
):
    df = read_store_df()
    if df.empty:
        return {"count": 0, "items": []}

    if "timestamp" in df.columns:
        df = df.assign(_ts=pd.to_datetime(df["timestamp"], errors="coerce")).sort_values("_ts", ascending=False).drop(columns=["_ts"])

    page_df = df.iloc[offset: offset + limit].copy()
    page_df = page_df.where(pd.notnull(page_df), None)

    return {
        "count": int(len(df)),
        "items": page_df.to_dict(orient="records"),
    }


@app.post("/ingest", response_model=PredictionResponse)
def ingest_email(payload: IngestEmailRequest):
    if not payload.body or len(payload.body.strip()) == 0:
        raise HTTPException(status_code=400, detail="Email body is required")

    full_text = f"{payload.subject}. {payload.body}".strip()
    cleaned = clean_text(full_text)

    if category_model is not None:
        try:
            technical_category = str(category_model.predict([cleaned])[0]).lower()
        except Exception:
            technical_category = category_rule_fallback(cleaned)
    else:
        technical_category = category_rule_fallback(cleaned)

    predicted_category = technical_category if technical_category in CATEGORY_MAP else category_rule_fallback(cleaned)

    signal_row = urgency_rule_score(cleaned, technical_category, urgency_keywords)
    if urgency_model is not None and tfidf_vectorizer is not None:
        try:
            X_single, signal_row = build_urgency_feature_vector(cleaned, technical_category, tfidf_vectorizer, urgency_keywords)
            ml_prob = get_ml_proba(urgency_model, X_single)[0]
            rule_prob = rule_prob_vector(signal_row)
            predicted_urgency = hybrid_predict_single(ml_prob, rule_prob, signal_row)
        except Exception:
            predicted_urgency = signal_row["rule_label"]
    else:
        predicted_urgency = signal_row["rule_label"]

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": payload.source,
        "subject": payload.subject or "(no subject)",
        "email_text": cleaned,
        "predicted_category": predicted_category,
        "predicted_urgency": predicted_urgency,
        "technical_category": technical_category,
    }

    append_store(result)

    route_payload = {
        "subject": result["subject"],
        "source": result["source"],
        "predicted_category": predicted_category,
        "predicted_urgency": predicted_urgency,
        "attachments": payload.attachments,
        "timestamp": result["timestamp"],
    }
    routed_systems, callback_sent = route_to_systems(route_payload, payload.target_systems, payload.callback_url)

    return {
        "timestamp": result["timestamp"],
        "source": result["source"],
        "subject": result["subject"],
        "predicted_category": predicted_category,
        "predicted_urgency": predicted_urgency,
        "technical_category": technical_category,
        "routed_systems": routed_systems,
        "callback_sent": callback_sent,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
