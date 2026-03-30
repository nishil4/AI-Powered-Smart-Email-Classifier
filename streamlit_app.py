import os
import re
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Smart Email Classifier Dashboard",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE2_MODEL_PATH = os.path.join(BASE_DIR, "module2_trained_models", "best_email_classifier.pkl")
MODULE3_MODEL_PATH = os.path.join(BASE_DIR, "module3_trained_models", "best_urgency_classifier.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "module3_trained_models", "tfidf_vectorizer.pkl")
RULE_CONFIG_PATH = os.path.join(BASE_DIR, "module3_trained_models", "urgency_rule_config.json")
STORE_DIR = os.path.join(BASE_DIR, "dashboard_data")
STORE_PATH = os.path.join(STORE_DIR, "email_predictions_log.csv")

CATEGORY_MAP = {
    "forum": "forum",
    "promotions": "promotions",
    "social_media": "social_media",
    "spam": "spam",
    "updates": "updates",
    "verify_code": "verify_code",
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

# API Configuration
API_URL = "https://ai-powered-smart-email-classifier-aek7.onrender.com"
ALLOW_LOCAL_CSV_FALLBACK = os.getenv("ALLOW_LOCAL_CSV_FALLBACK", "false").strip().lower() == "true"

ALL_CATEGORIES = ["forum", "promotions", "social_media", "spam", "updates", "verify_code"]
ALL_URGENCIES = ["high", "medium", "low"]


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


@st.cache_resource(show_spinner=False)
def load_models():
    category_model = None
    urgency_model = None
    tfidf_vectorizer = None

    if os.path.exists(MODULE2_MODEL_PATH):
        category_model = joblib.load(MODULE2_MODEL_PATH)
    if os.path.exists(MODULE3_MODEL_PATH):
        urgency_model = joblib.load(MODULE3_MODEL_PATH)
    if os.path.exists(TFIDF_PATH):
        tfidf_vectorizer = joblib.load(TFIDF_PATH)

    return category_model, urgency_model, tfidf_vectorizer


@st.cache_data(show_spinner=False)
def load_rule_keywords():
    if os.path.exists(RULE_CONFIG_PATH):
        return pd.read_json(RULE_CONFIG_PATH, typ="series").to_dict().get("urgency_keywords", {})
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
    verify_code_terms = [
        "verification code", "verify code", "otp", "pin", "passcode", "authentication code", "access code",
        "account verification", "password reset"
    ]
    spam_terms = ["click here", "limited time", "winner", "unsubscribe", "scam", "fake", "claim prize"]
    promotion_terms = ["offer", "free", "discount", "sale", "deal", "coupon", "clearance", "exclusive"]
    social_terms = ["like", "comment", "follow", "friend request", "event invitation", "rsvp", "group"]
    forum_terms = ["thread", "wiki", "moderator", "forum", "discussion", "post moved", "reply"]

    if any(k in t for k in verify_code_terms):
        return "verify_code"
    if any(k in t for k in spam_terms):
        return "spam"
    if any(k in t for k in promotion_terms):
        return "promotions"
    if any(k in t for k in social_terms):
        return "social_media"
    if any(k in t for k in forum_terms):
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
        "exclamations": exclamations,
        "has_numeric_code": has_numeric_code,
        "caps_ratio": caps_ratio,
        "text_length": text_length,
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


def build_urgency_feature_vector(text: str, category_raw: str, tfidf_vectorizer, urgency_keywords: dict):
    signal_row = urgency_rule_score(text, category_raw, urgency_keywords)
    signal_df = pd.DataFrame([signal_row])

    category_feature_row = {k: 0 for k in CATEGORY_FEATURES}
    cat_key = f"category_{str(category_raw).lower()}"
    if cat_key in category_feature_row:
        category_feature_row[cat_key] = 1

    signal_features = signal_df[RULE_SIGNAL_COLS].astype(float)
    cat_features = pd.DataFrame([category_feature_row]).astype(float)
    tabular = pd.concat([signal_features.reset_index(drop=True), cat_features.reset_index(drop=True)], axis=1)

    tfidf_vec = tfidf_vectorizer.transform([text]).toarray()
    combined = np.hstack([tfidf_vec, tabular.to_numpy(dtype=float)])
    return combined, signal_row


def predict_category(text: str, category_model):
    if category_model is not None:
        try:
            pred = category_model.predict([text])[0]
            return str(pred).lower()
        except Exception:
            pass
    return category_rule_fallback(text)


def predict_urgency(text: str, category_raw: str, urgency_model, tfidf_vectorizer, urgency_keywords: dict):
    signal_row = urgency_rule_score(text, category_raw, urgency_keywords)

    if urgency_model is not None and tfidf_vectorizer is not None:
        try:
            X_single, signal_row = build_urgency_feature_vector(text, category_raw, tfidf_vectorizer, urgency_keywords)
            ml_prob = get_ml_proba(urgency_model, X_single)[0]
            rule_prob = rule_prob_vector(signal_row)
            return hybrid_predict_single(ml_prob, rule_prob, signal_row)
        except Exception:
            pass

    return signal_row["rule_label"]


def ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)
    if not os.path.exists(STORE_PATH):
        cols = [
            "timestamp",
            "source",
            "subject",
            "email_text",
            "predicted_category",
            "predicted_urgency",
            "technical_category",
        ]
        pd.DataFrame(columns=cols).to_csv(STORE_PATH, index=False)


def load_live_data():
    ensure_store()
    base_cols = ["timestamp", "source", "subject", "email_text", "predicted_category", "predicted_urgency", "technical_category"]

    # Primary source: integration API (Render), where Gmail ingestion writes records.
    try:
        response = requests.get(f"{API_URL}/predictions", params={"limit": 5000}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", [])
        df = pd.DataFrame(items) if items else pd.DataFrame(columns=base_cols)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        st.session_state["api_fetch_error"] = False
        return df
    except Exception:
        st.session_state["api_fetch_error"] = True

    if not ALLOW_LOCAL_CSV_FALLBACK:
        st.warning("Live API data is currently unavailable. Manual local rows are hidden to avoid mixed data.")
        return pd.DataFrame(columns=base_cols)

    # Fallback source: local CSV (for local/manual testing).
    try:
        df = pd.read_csv(STORE_PATH)
        if len(df) > 0:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=base_cols)


def append_prediction(record: dict):
    ensure_store()
    df = load_live_data()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(STORE_PATH, index=False)


def apply_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    selected_categories = st.sidebar.multiselect(
        "Filter 1: Category Filter",
        options=ALL_CATEGORIES,
        default=ALL_CATEGORIES,
    )
    selected_urgencies = st.sidebar.multiselect(
        "Filter 2: Urgency Filter",
        options=ALL_URGENCIES,
        default=ALL_URGENCIES,
    )

    if len(df) > 0 and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        min_date = ts.min().date() if not ts.isna().all() else datetime.now().date() - timedelta(days=365)
        max_date = ts.max().date() if not ts.isna().all() else datetime.now().date()
    else:
        min_date = datetime.now().date() - timedelta(days=365)
        max_date = datetime.now().date()

    date_range = st.sidebar.date_input("Filter 3: Date Filter", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    out = df.copy()
    out["timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out[out["predicted_category"].isin(selected_categories)]
    out = out[out["predicted_urgency"].isin(selected_urgencies)]
    out = out[(out["timestamp_dt"].dt.date >= start_date) & (out["timestamp_dt"].dt.date <= end_date)]
    return out.drop(columns=["timestamp_dt"], errors="ignore")





def render_heatmap(df: pd.DataFrame):
    """Render category vs urgency heatmap"""
    if df.empty:
        st.info("No data available for heatmap.")
        return
    
    heatmap_data = pd.crosstab(
        df["predicted_category"],
        df["predicted_urgency"],
        margins=False
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale="Viridis",
        text=heatmap_data.values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig.update_layout(title="Category vs Urgency Heatmap", xaxis_title="Urgency", yaxis_title="Category")
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(df: pd.DataFrame):
    """Render confusion matrix for category predictions"""
    if df.empty:
        st.info("No data available for confusion matrix.")
        return
    
    categories = sorted(df["predicted_category"].unique())
    if len(categories) < 2:
        st.info("Need at least 2 categories for confusion matrix.")
        return
    
    cm = pd.crosstab(
        df["predicted_category"],
        df["predicted_category"],
        rownames=["True"],
        colnames=["Predicted"]
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm.values,
        x=cm.columns,
        y=cm.index,
        colorscale="Blues",
        text=cm.values,
        texttemplate="%{text}",
        textfont={"size": 11}
    ))
    fig.update_layout(title="Category Prediction Distribution", xaxis_title="Category", yaxis_title="Category")
    st.plotly_chart(fig, use_container_width=True)


def render_advanced_visualizations(df: pd.DataFrame):
    """Render advanced charts including heatmap, line chart, and distribution"""
    if df.empty:
        st.info("No data available for advanced visualizations.")
        return
    
    aviz_col1, aviz_col2 = st.columns(2)
    
    with aviz_col1:
        st.markdown("### Category vs Urgency Heatmap")
        render_heatmap(df)
    
    with aviz_col2:
        st.markdown("### Category Distribution Heatmap")
        render_confusion_matrix(df)
    
    # Time series line chart for both categories and urgencies
    st.markdown("### Email Volume Trends by Category")
    trend_df = df.copy()
    trend_df["date"] = pd.to_datetime(trend_df["timestamp"], errors="coerce").dt.date
    
    category_trend = trend_df.groupby(["date", "predicted_category"], as_index=False).size()
    fig_cat_trend = px.line(
        category_trend,
        x="date",
        y="size",
        color="predicted_category",
        markers=True,
        title="Email Volume by Category Over Time",
        labels={"size": "Count", "date": "Date", "predicted_category": "Category"}
    )
    st.plotly_chart(fig_cat_trend, use_container_width=True)
    
    # Urgency trend line chart
    st.markdown("### Email Volume Trends by Urgency")
    urgency_trend = trend_df.groupby(["date", "predicted_urgency"], as_index=False).size()
    fig_urg_trend = px.line(
        urgency_trend,
        x="date",
        y="size",
        color="predicted_urgency",
        markers=True,
        title="Email Volume by Urgency Over Time",
        labels={"size": "Count", "date": "Date", "predicted_urgency": "Urgency"},
        category_orders={"predicted_urgency": ["high", "medium", "low"]}
    )
    st.plotly_chart(fig_urg_trend, use_container_width=True)


def render_detailed_analysis(df: pd.DataFrame):
    """Render detailed statistics and analysis"""
    if df.empty:
        st.info("No data available for detailed analysis.")
        return
    
    st.markdown("### Email Statistics")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Emails", f"{len(df):,}")
    col2.metric("Unique Categories", df["predicted_category"].nunique())
    col3.metric("Avg Emails/Day", f"{len(df) / max(1, (pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()).days + 1):.1f}")
    col4.metric("Unique Sources", df["source"].nunique())
    
    st.divider()
    
    # Detailed breakdown tables
    st.markdown("### Category Breakdown")
    cat_stats = df.groupby("predicted_category").agg({
        "subject": "count",
        "predicted_urgency": lambda x: (x == "high").sum()
    }).rename(columns={"subject": "Total", "predicted_urgency": "High Urgency"})
    st.dataframe(cat_stats, use_container_width=True)
    
    st.markdown("### Urgency Breakdown")
    urg_stats = df.groupby("predicted_urgency").agg({
        "subject": "count"
    }).rename(columns={"subject": "Count"})
    st.dataframe(urg_stats, use_container_width=True)

def render_analytics(df: pd.DataFrame):
    """Render quick analytics with key metrics and charts"""
    if df.empty:
        st.info("No data available. Classifications will appear here.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total emails", f"{len(df):,}")
    col2.metric("High urgency", int((df["predicted_urgency"] == "high").sum()))
    col3.metric("Spam category", int((df["predicted_category"] == "spam").sum()))

    col_left, col_right = st.columns(2)

    cat_counts = df["predicted_category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    fig_cat = px.bar(cat_counts, x="category", y="count", title="Emails by Category", color="category")
    col_left.plotly_chart(fig_cat, use_container_width=True)

    urg_counts = df["predicted_urgency"].value_counts().reset_index()
    urg_counts.columns = ["urgency", "count"]
    fig_urg = px.pie(urg_counts, names="urgency", values="count", title="Urgency Distribution",
                     color_discrete_map={"high": "#FF6B6B", "medium": "#FFA500", "low": "#4ECDC4"})
    col_right.plotly_chart(fig_urg, use_container_width=True)


def process_csv_batch(df: pd.DataFrame, text_col: str, category_model, urgency_model, tfidf_vectorizer, urgency_keywords: dict) -> pd.DataFrame:
    """Classify all rows in a dataframe"""
    results = []
    progress = st.progress(0, text="Processing emails...")
    total = len(df)
    for i, row in df.iterrows():
        raw_text = str(row.get(text_col, ""))
        cleaned = clean_text(raw_text)
        technical_category = predict_category(cleaned, category_model)
        dataset_category = technical_category if technical_category in CATEGORY_MAP else category_rule_fallback(cleaned)
        predicted_urgency = predict_urgency(cleaned, technical_category, urgency_model, tfidf_vectorizer, urgency_keywords)
        results.append({
            "original_text": raw_text[:120] + ("..." if len(raw_text) > 120 else ""),
            "predicted_category": dataset_category,
            "predicted_urgency": predicted_urgency,
        })
        progress.progress(min(int((i + 1) / total * 100), 100), text=f"Processing: {i + 1}/{total}")
    progress.empty()
    out = df.copy().reset_index(drop=True)
    result_df = pd.DataFrame(results)
    out["predicted_category"] = result_df["predicted_category"]
    out["predicted_urgency"] = result_df["predicted_urgency"]
    return out

def main():
    st.title("AI Powered Smart Email Classifier")
    st.caption("Real-time email classification, urgency detection, and advanced analytics")
        # Disable automatic page refresh
        # st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        if st.button("Refresh Now", use_container_width=True):
                import streamlit.components.v1 as components
                components.html(
                    """
                    <script>
                    const u = new URL(window.parent.location.href);
                    u.searchParams.set('refreshTs', Date.now().toString());
                    window.parent.location.href = u.toString();
                    </script>
                    """,
                    height=0,
                )
    with col3:
        st.metric("Status", "Live")

    category_model, urgency_model, tfidf_vectorizer = load_models()
    urgency_keywords = load_rule_keywords()
    
    live_df = load_live_data()
    data_df = live_df.copy()
    
    if len(data_df) > 0:
        data_df["predicted_urgency"] = data_df["predicted_urgency"].astype(str).str.lower()
        data_df["predicted_category"] = data_df["predicted_category"].astype(str).str.lower()

    filtered_df = apply_filters(data_df)
    
    if len(filtered_df) > 0:
        st.success(f"Loaded {len(filtered_df)} classifications")
    else:
        st.info("Awaiting Gmail classifications from Apps Script...")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Dashboard",
        "Analysis",
        "Advanced Visualization",
        "Detailed Stats",
        "Batch Processing"
    ])

    # ========== TAB 1: LIVE DASHBOARD ==========
    with tab1:
        st.subheader("Live Email Classifier")
        st.markdown("Classify emails in real-time. Emails are automatically ingested from Gmail every 5 minutes.")
        
        with st.form("classify_form", clear_on_submit=False):
            source = "Gmail"
            col_source, col_empty = st.columns([1, 4])
            with col_source:
                st.text_input("Source", value="Gmail", disabled=True)
            
            subject = st.text_input("Subject", placeholder="e.g., Payment failed after update")
            body = st.text_area("Email Body", height=150, placeholder="Paste email content here...")
            submit = st.form_submit_button("Classify Email", use_container_width=True)

        if submit:
            full_text = f"{subject}. {body}".strip()
            if len(full_text.strip()) < 3:
                st.warning("Please add a subject or body before classifying.")
            else:
                with st.spinner("Classifying..."):
                    api_payload = {
                        "source": "Dashboard",
                        "subject": subject if subject else "(no subject)",
                        "body": body,
                        "attachments": [],
                        "target_systems": [],
                        "callback_url": None,
                    }

                    try:
                        response = requests.post(f"{API_URL}/ingest", json=api_payload, timeout=20)
                        response.raise_for_status()
                        result = response.json()
                    except Exception as e:
                        st.error(f"Classification failed via API: {e}")
                        result = None

                if result is not None:
                    st.success("Classification complete!")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Category", str(result.get("predicted_category", "-")).upper())
                    col_b.metric("Urgency", str(result.get("predicted_urgency", "-")).upper())
                    col_c.metric("Source", str(result.get("source", "Dashboard")))

        st.divider()
        render_analytics(filtered_df)

    # ========== TAB 2: ANALYSIS ==========
    with tab2:
        st.subheader("Analytics Overview")
        
        if filtered_df.empty:
            st.info("No data to analyze. Run classifications or upload data to populate.")
        else:
            analysis_df = filtered_df.copy()
            analysis_df["timestamp_dt"] = pd.to_datetime(analysis_df["timestamp"], errors="coerce")

            # KPI row focused on trend and density
            total_count = len(analysis_df)
            high_count = int((analysis_df["predicted_urgency"] == "high").sum())
            high_pct = (high_count / total_count * 100.0) if total_count else 0.0
            active_categories = int(analysis_df["predicted_category"].nunique())
            valid_ts = analysis_df["timestamp_dt"].dropna()
            span_days = max(1, ((valid_ts.max() - valid_ts.min()).days + 1) if not valid_ts.empty else 1)
            avg_per_day = total_count / span_days

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Emails", f"{total_count:,}")
            col2.metric("High Urgency Rate", f"{high_pct:.1f}%")
            col3.metric("Active Categories", active_categories)
            col4.metric("Avg per Day", f"{avg_per_day:.1f}")

            st.divider()

            # 1) Urgency composition by category (100% stacked)
            composition = (
                analysis_df.groupby(["predicted_category", "predicted_urgency"])
                .size()
                .reset_index(name="count")
            )
            if not composition.empty:
                fig_comp = px.bar(
                    composition,
                    x="predicted_category",
                    y="count",
                    color="predicted_urgency",
                    title="Urgency Composition by Category",
                    labels={
                        "predicted_category": "Category",
                        "count": "Email Count",
                        "predicted_urgency": "Urgency",
                    },
                    barmode="relative",
                    color_discrete_map={"high": "#E15759", "medium": "#F28E2B", "low": "#4E79A7"},
                )
                fig_comp.update_layout(legend_title_text="Urgency")
                st.plotly_chart(fig_comp, use_container_width=True)

            # 2) High-urgency leaderboard by category
            high_by_cat = (
                analysis_df.assign(is_high=analysis_df["predicted_urgency"].eq("high"))
                .groupby("predicted_category", as_index=False)
                .agg(total=("subject", "count"), high_count=("is_high", "sum"))
            )
            if not high_by_cat.empty:
                high_by_cat["high_rate"] = (high_by_cat["high_count"] / high_by_cat["total"]).fillna(0.0)
                high_by_cat = high_by_cat.sort_values(["high_count", "high_rate"], ascending=[False, False])

                fig_risk = px.bar(
                    high_by_cat,
                    x="predicted_category",
                    y="high_count",
                    color="high_rate",
                    title="High-Urgency Load by Category",
                    labels={"predicted_category": "Category", "high_count": "High-Urgency Emails", "high_rate": "High Rate"},
                    color_continuous_scale="Reds",
                    text="high_count",
                )
                fig_risk.update_traces(textposition="outside")
                st.plotly_chart(fig_risk, use_container_width=True)

            # 3) Time-pattern analysis
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday_counts = (
                    analysis_df.dropna(subset=["timestamp_dt"])
                    .assign(weekday=lambda d: d["timestamp_dt"].dt.day_name())
                    .groupby("weekday")
                    .size()
                    .reindex(weekday_order, fill_value=0)
                    .reset_index(name="count")
                )
                fig_weekday = px.bar(
                    weekday_counts,
                    x="weekday",
                    y="count",
                    title="Emails by Day of Week",
                    labels={"weekday": "Day", "count": "Email Count"},
                    color="count",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_weekday, use_container_width=True)

            with pcol2:
                hour_counts = (
                    analysis_df.dropna(subset=["timestamp_dt"])
                    .assign(hour=lambda d: d["timestamp_dt"].dt.hour)
                    .groupby("hour")
                    .size()
                    .reindex(range(24), fill_value=0)
                    .reset_index(name="count")
                )
                fig_hour = px.line(
                    hour_counts,
                    x="hour",
                    y="count",
                    markers=True,
                    title="Hourly Email Arrival Pattern",
                    labels={"hour": "Hour of Day", "count": "Email Count"},
                )
                st.plotly_chart(fig_hour, use_container_width=True)

    # ========== TAB 3: ADVANCED VISUALIZATION ==========
    with tab3:
        st.subheader("Advanced Data Visualization")
        
        if filtered_df.empty:
            st.info("No data to visualize. Run classifications or upload data first.")
        else:
            render_advanced_visualizations(filtered_df)

    # ========== TAB 4: DETAILED STATS ==========
    with tab4:
        st.subheader("Detailed Statistics & Insights")
        
        if filtered_df.empty:
            st.info("No data available. Run classifications or upload data first.")
        else:
            render_detailed_analysis(filtered_df)
            
            st.divider()
            st.markdown("### Recent Classifications")
            display_cols = ["timestamp", "source", "subject", "predicted_category", "predicted_urgency"]
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(
                filtered_df[display_cols].tail(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Download full filtered results
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download All Filtered Results (CSV)",
                data=csv_data,
                file_name=f"email_classifications_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ========== TAB 5: BATCH PROCESSING ==========
    with tab5:
        st.subheader("Batch Process CSV Data")
        st.markdown("Upload a CSV file containing email text and automatically classify all rows.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                return

            if upload_df.empty:
                st.warning("The file is empty.")
                return

            st.markdown(f"**Loaded {len(upload_df):,} rows**")
            st.dataframe(upload_df.head(5), use_container_width=True, hide_index=True)

            # Select text column
            text_columns = upload_df.columns.tolist()
            guessed = next(
                (c for c in text_columns if c.lower() in ["text", "body", "email", "message", "content", "email_text"]),
                text_columns[0]
            )
            text_col = st.selectbox("Which column contains email text?", options=text_columns, index=text_columns.index(guessed))

            if st.button("Classify All Rows", use_container_width=True):
                with st.spinner("Processing..."):
                    result_df = process_csv_batch(
                        upload_df,
                        text_col,
                        category_model,
                        urgency_model,
                        tfidf_vectorizer,
                        urgency_keywords
                    )

                st.success(f"Processed {len(result_df):,} rows successfully!")

                # Results metrics
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Total Rows", f"{len(result_df):,}")
                res_col2.metric("High Urgency", int((result_df["predicted_urgency"] == "high").sum()))
                res_col3.metric("Medium Urgency", int((result_df["predicted_urgency"] == "medium").sum()))
                res_col4.metric("Low Urgency", int((result_df["predicted_urgency"] == "low").sum()))

                st.divider()

                # Results visualization
                rescol1, rescol2 = st.columns(2)
                
                with rescol1:
                    cat_counts = result_df["predicted_category"].value_counts().reset_index()
                    cat_counts.columns = ["category", "count"]
                    fig = px.bar(cat_counts, x="category", y="count", title="Category Distribution", color="category")
                    st.plotly_chart(fig, use_container_width=True)

                with rescol2:
                    urg_counts = result_df["predicted_urgency"].value_counts().reset_index()
                    urg_counts.columns = ["urgency", "count"]
                    fig = px.pie(urg_counts, names="urgency", values="count", title="Urgency Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                # Display full results
                st.markdown("### Classification Results")
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # Download button
                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_out,
                    file_name=f"classified_{uploaded_file.name}",
                    mime="text/csv",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
