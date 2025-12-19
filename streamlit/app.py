# streamlit/app.py
import os
import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Drug Effectiveness Predictor",
    page_icon="💊",
    layout="wide",
)

# -----------------------------
# API config
# -----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"
MODEL_INFO_URL = f"{API_URL}/model-info"

# -----------------------------
# Example inputs (great for demo)
# -----------------------------
EXAMPLES = {
    "🌟 Highly Effective (Positive)": {
        "urlDrugName": "Prozac",
        "condition": "Depression",
        "rating": 9.0,
        "sideEffects": "Mild Side Effects",
        "benefitsReview": (
            "This medication has been extremely effective for my depression. "
            "My mood improved significantly, I feel more motivated and focused, "
            "and my daily functioning has greatly improved. The positive effects "
            "were noticeable within a few weeks and have remained consistent."
        ),
        "sideEffectsReview": (
            "I experienced very mild nausea during the first few days, but it quickly "
            "went away and did not interfere with my daily activities."
        ),
        "commentsReview": (
            "Overall this drug has been life changing for me. I am very satisfied with "
            "the results and would strongly recommend it to others under medical supervision."
        ),
    },
    "😐 Moderately Effective (Mixed)": {
        "urlDrugName": "Metformin",
        "condition": "Type 2 Diabetes",
        "rating": 6.0,
        "sideEffects": "Moderate Side Effects",
        "benefitsReview": (
            "It helped bring my blood sugar down a bit, but results were not consistent "
            "every day. I did notice some improvement after a few weeks."
        ),
        "sideEffectsReview": (
            "Some stomach upset and nausea especially early on, but it became more manageable."
        ),
        "commentsReview": (
            "It works okay, not amazing. I will continue for now but may adjust treatment "
            "with my doctor."
        ),
    },
    "❌ Ineffective (Negative)": {
        "urlDrugName": "Ambien",
        "condition": "Insomnia",
        "rating": 2.5,
        "sideEffects": "Severe Side Effects",
        "benefitsReview": (
            "It barely helped me sleep and the effect wore off quickly."
        ),
        "sideEffectsReview": (
            "Severe next-day grogginess, confusion, and memory problems. I also felt anxious."
        ),
        "commentsReview": (
            "Overall ineffective for me and the side effects were not worth it. I stopped using it."
        ),
    },
}

SIDE_EFFECTS_OPTIONS = [
    "No Side Effects",
    "Mild Side Effects",
    "Moderate Side Effects",
    "Severe Side Effects",
    "Extremely Severe Side Effects",
]

EMOJI = {
    "Ineffective": "❌",
    "Marginally Effective": "⚠️",
    "Moderately Effective": "✅",
    "Considerably Effective": "👍",
    "Highly Effective": "🌟",
}

# -----------------------------
# Session state init
# -----------------------------
def init_state():
    if "urlDrugName" not in st.session_state:
        st.session_state.update(EXAMPLES["🌟 Highly Effective (Positive)"])

init_state()

def load_example(name: str):
    st.session_state.update(EXAMPLES[name])
    st.rerun()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("💊 Drug Review Classifier")

    st.markdown("### 📚 Demo Examples")
    for ex_name in EXAMPLES.keys():
        if st.button(ex_name, use_container_width=True):
            load_example(ex_name)

    st.markdown("---")
    st.markdown("### 🔗 API Status")
    try:
        h = requests.get(HEALTH_URL, timeout=3).json()
        st.success(f"API: {h.get('status', 'ok')}")
        st.write(f"Model loaded: `{h.get('model_loaded')}`")
    except Exception:
        st.error("API not reachable")
        st.caption("Start API first: `uvicorn api.app:app --reload`")

    with st.expander("ℹ️ Model Info"):
        try:
            mi = requests.get(MODEL_INFO_URL, timeout=3).json()
            st.json(mi)
        except Exception:
            st.write("Unable to fetch model info.")

# -----------------------------
# Main content
# -----------------------------
st.title("Drug Effectiveness Prediction (Text Classification)")
st.caption("Predict effectiveness class from patient review text (TF-IDF + ML).")

# Explain app (great for grading / demo)
with st.expander("📌 What is this app? (Project Summary)", expanded=True):
    st.markdown(
        """
**Goal:** Classify a patient drug review into one of five effectiveness categories:

- Ineffective  
- Marginally Effective  
- Moderately Effective  
- Considerably Effective  
- Highly Effective  

**How it works (high level):**
1. User enters review text (benefits, side effects, additional comments) and context fields.
2. The Streamlit UI sends the input to a FastAPI backend (`/predict`).
3. The backend runs a trained ML model and returns:
   - Predicted effectiveness class  
   - Optional class probabilities (confidence)

**Important note:** In your current model version, predictions are primarily driven by the **review text**.
"""
    )

# -----------------------------
# Input form
# -----------------------------
st.markdown("## 📝 Enter Review Details")

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])

with c1:
    urlDrugName = st.text_input("Drug Name", key="urlDrugName", placeholder="e.g., Lipitor")
with c2:
    condition = st.text_input("Medical Condition", key="condition", placeholder="e.g., High Cholesterol")
with c3:
    rating = st.slider("Overall Rating (1–10)", 1.0, 10.0, float(st.session_state.get("rating", 7.0)), 0.5, key="rating")
with c4:
    sideEffects = st.selectbox("Side Effects Severity", SIDE_EFFECTS_OPTIONS,
                               index=SIDE_EFFECTS_OPTIONS.index(st.session_state.get("sideEffects", "Moderate Side Effects")),
                               key="sideEffects")

st.markdown("### 🧾 Review Text (Most important)")
benefitsReview = st.text_area("Benefits Review", key="benefitsReview", height=120,
                              placeholder="Describe benefits and improvements...")
sideEffectsReview = st.text_area("Side Effects Review", key="sideEffectsReview", height=120,
                                 placeholder="Describe side effects and severity...")
commentsReview = st.text_area("Additional Comments", key="commentsReview", height=120,
                              placeholder="Anything else about the experience...")

# Validation hints
st.info(
    "Tip: Use detailed sentences (not single words). The model is text-based, so richer descriptions usually produce more reliable predictions."
)

# -----------------------------
# Predict button
# -----------------------------
colA, colB = st.columns([1, 2])

with colA:
    predict_btn = st.button("🔮 Predict Effectiveness", type="primary", use_container_width=True)

with colB:
    st.caption(f"Backend: `{PREDICT_URL}`")

if predict_btn:
    # Basic validation (avoid empty payloads)
    if not benefitsReview.strip() and not sideEffectsReview.strip() and not commentsReview.strip():
        st.error("Please enter at least some review text (Benefits / Side Effects / Comments).")
    else:
        payload = {
            "instances": [{
                "urlDrugName": urlDrugName,
                "condition": condition,
                "rating": float(rating),
                "sideEffects": sideEffects,
                "benefitsReview": benefitsReview,
                "sideEffectsReview": sideEffectsReview,
                "commentsReview": commentsReview,
            }]
        }

        with st.spinner("Sending to API and predicting..."):
            try:
                res = requests.post(PREDICT_URL, json=payload, timeout=30)
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                if res.status_code != 200:
                    st.error(f"❌ API error: HTTP {res.status_code} - {res.text}")
                else:
                    data = res.json()
                    pred = (data.get("predictions") or ["Unknown"])[0]
                    probs = data.get("probabilities")

                    st.markdown("## ✅ Prediction Result")
                    emoji = EMOJI.get(pred, "💊")
                    st.metric("Predicted Effectiveness", f"{emoji} {pred}")

                    if probs:
                        st.markdown("### 📊 Confidence (Class Probabilities)")
                        p0 = probs[0]
                        dfp = pd.DataFrame(
                            [{"Class": k, "Probability": float(v)} for k, v in p0.items()]
                        ).sort_values("Probability", ascending=False)
                        st.dataframe(dfp, use_container_width=True, hide_index=True)
                        st.bar_chart(dfp.set_index("Class")["Probability"])

                    with st.expander("🧾 View submitted payload"):
                        st.json(payload)

# Footer
st.markdown("---")
st.caption(
    "Built with Streamlit + FastAPI. Model trained on drug review text for effectiveness classification."
)
