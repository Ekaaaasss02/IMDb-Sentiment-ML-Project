import json
from pathlib import Path

import joblib
import streamlit as st

from src.preprocessing import clean_text


ARTIFACTS = Path("artifacts")
BEST_JSON = ARTIFACTS / "best_model.json"


@st.cache_resource
def load_model_and_vectorizer():
    if not BEST_JSON.exists():
        return None, None, None

    with open(BEST_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = joblib.load(meta["model_path"])
    vectorizer = joblib.load(meta["vectorizer_path"])
    return model, vectorizer, meta


st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

st.title("IMDb Sentiment Analyzer")
st.write("Machine-learning based positive/negative movie-review classification.")

model, vectorizer, meta = load_model_and_vectorizer()

if model is None:
    st.warning(
        "No trained model found. Train the project first:\n\n"
        "`python src/train.py --data data/IMDB_Dataset.csv --output artifacts`"
    )
    st.stop()

st.caption(
    f"Loaded model: {meta['model']} + {meta['vectorization']} | "
    f"reported local test accuracy: {meta['accuracy']:.2%}"
)

review = st.text_area(
    "Enter a movie review",
    height=180,
    placeholder="Example: The acting was brilliant and the story was fantastic...",
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.error("Please enter a review.")
    else:
        cleaned = clean_text(review)
        X = vectorizer.transform([cleaned])

        prediction = int(model.predict(X)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0][prediction])
            confidence = probability * 100
        else:
            confidence = None

        if prediction == 1:
            st.success(" Positive Review")
        else:
            st.error(" Negative Review")

        if confidence is not None:
            st.metric("Model confidence", f"{confidence:.2f}%")

        with st.expander("See preprocessing"):
            st.write(cleaned)
