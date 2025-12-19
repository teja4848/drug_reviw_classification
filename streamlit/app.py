import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Drug Effectiveness Prediction",
    page_icon="💊",
    layout="centered"
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCHEMA_PATH = Path("/app/data/data_schema.json")

# API_URL is set in docker-compose environment
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
MODEL_INFO_ENDPOINT = f"{API_BASE_URL}/model-info"

# -----------------------------------------------------------------------------
# Load schema from JSON file
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        # Return default schema if file not found
        return {
            "numerical": {
                "rating": {"min": 1.0, "max": 10.0, "mean": 6.5, "median": 7.0}
            },
            "categorical": {
                "urlDrugName": {"unique_values": [], "value_counts": {}},
                "condition": {"unique_values": [], "value_counts": {}},
                "sideEffects": {
                    "unique_values": [
                        "No Side Effects",
                        "Mild Side Effects",
                        "Moderate Side Effects",
                        "Severe Side Effects",
                        "Extremely Severe Side Effects"
                    ],
                    "value_counts": {}
                }
            }
        }
    with open(path, "r") as f:
        return json.load(f)


schema = load_schema(SCHEMA_PATH)

numerical_features = schema.get("numerical", {})
categorical_features = schema.get("categorical", {})

# -----------------------------------------------------------------------------
# Effectiveness classes (target variable)
# -----------------------------------------------------------------------------
EFFECTIVENESS_CLASSES = [
    "Ineffective",
    "Marginally Effective",
    "Moderately Effective",
    "Considerably Effective",
    "Highly Effective",
]

SIDE_EFFECTS_OPTIONS = [
    "No Side Effects",
    "Mild Side Effects",
    "Moderate Side Effects",
    "Severe Side Effects",
    "Extremely Severe Side Effects",
]

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("💊 Drug Effectiveness Prediction")
st.write(
    f"This app predicts drug effectiveness based on patient reviews. "
    f"Enter the review details below and click **Predict**."
)

st.header("📝 Drug Review Input")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Drug and Condition Selection
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    # Drug name - either from schema or free text
    drug_options = categorical_features.get("urlDrugName", {}).get("unique_values", [])
    if drug_options:
        user_input["urlDrugName"] = st.selectbox(
            "Drug Name",
            options=drug_options,
            help="Select the drug being reviewed"
        )
    else:
        user_input["urlDrugName"] = st.text_input(
            "Drug Name",
            value="Lipitor",
            help="Enter the drug name"
        )

with col2:
    # Condition - either from schema or free text
    condition_options = categorical_features.get("condition", {}).get("unique_values", [])
    if condition_options:
        user_input["condition"] = st.selectbox(
            "Medical Condition",
            options=condition_options,
            help="Select the condition being treated"
        )
    else:
        user_input["condition"] = st.text_input(
            "Medical Condition",
            value="High Cholesterol",
            help="Enter the medical condition"
        )

# -----------------------------------------------------------------------------
# Rating and Side Effects
# -----------------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    rating_stats = numerical_features.get("rating", {})
    user_input["rating"] = st.slider(
        "Overall Rating",
        min_value=1.0,
        max_value=10.0,
        value=float(rating_stats.get("median", 7.0)),
        step=0.5,
        help="Rate the drug from 1 (worst) to 10 (best)"
    )

with col4:
    user_input["sideEffects"] = st.selectbox(
        "Side Effects Severity",
        options=SIDE_EFFECTS_OPTIONS,
        index=2,  # Default to "Moderate Side Effects"
        help="Select the severity of side effects experienced"
    )

# -----------------------------------------------------------------------------
# Review Text Fields
# -----------------------------------------------------------------------------
st.subheader("📄 Review Text")

user_input["benefitsReview"] = st.text_area(
    "Benefits Review",
    value="",
    height=100,
    placeholder="Describe the benefits you experienced from this medication...",
    help="What positive effects did you notice?"
)

user_input["sideEffectsReview"] = st.text_area(
    "Side Effects Review",
    value="",
    height=100,
    placeholder="Describe any side effects you experienced...",
    help="What negative effects or side effects did you notice?"
)

user_input["commentsReview"] = st.text_area(
    "Additional Comments",
    value="",
    height=100,
    placeholder="Any other comments about your experience...",
    help="Any additional information about your experience with this drug"
)

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict Button
# -----------------------------------------------------------------------------
if st.button("🔮 Predict Effectiveness", type="primary"):
    # Validate inputs
    if not user_input["urlDrugName"].strip():
        st.error("❌ Please enter a drug name.")
    elif not user_input["condition"].strip():
        st.error("❌ Please enter a medical condition.")
    else:
        payload = {"instances": [user_input]}

        with st.spinner("Analyzing review and predicting effectiveness..."):
            try:
                resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=30)
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request to API failed: {e}")
            else:
                if resp.status_code != 200:
                    st.error(f"❌ API error: HTTP {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    preds = data.get("predictions", [])
                    probs = data.get("probabilities", [])

                    if not preds:
                        st.warning("⚠️ No predictions returned from API.")
                    else:
                        pred = preds[0]
                        st.success("✅ Prediction complete!")

                        st.subheader("🎯 Prediction Result")

                        # Display prediction with nice formatting
                        effectiveness_emoji = {
                            "Ineffective": "❌",
                            "Marginally Effective": "⚠️",
                            "Moderately Effective": "✅",
                            "Considerably Effective": "👍",
                            "Highly Effective": "🌟",
                        }
                        emoji = effectiveness_emoji.get(pred, "💊")
                        st.metric(
                            label="Predicted Effectiveness",
                            value=f"{emoji} {pred}"
                        )

                        # Show probabilities if available
                        if probs:
                            st.subheader("📊 Confidence Scores")
                            prob_dict = probs[0]
                            
                            # Create a bar chart of probabilities
                            import pandas as pd
                            prob_df = pd.DataFrame([
                                {"Class": k, "Probability": v}
                                for k, v in prob_dict.items()
                            ])
                            prob_df = prob_df.sort_values("Probability", ascending=True)
                            
                            st.bar_chart(
                                prob_df.set_index("Class")["Probability"],
                                horizontal=True,
                            )

                        # Show input summary in expander
                        with st.expander("📋 View Input Summary"):
                            st.json(user_input)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"📁 Schema: `{SCHEMA_PATH}`  \n"
    f"🌐 API: `{API_BASE_URL}`"
)

# -----------------------------------------------------------------------------
# Sidebar - Example Reviews
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Example Reviews")
    st.write("Click to fill in example data:")
    
    if st.button("👍 Positive Review Example"):
        st.session_state["example"] = {
            "urlDrugName": "Prozac",
            "condition": "Depression",
            "rating": 9.0,
            "sideEffects": "Mild Side Effects",
            "benefitsReview": "This medication has significantly improved my mood and overall quality of life. I feel more motivated and engaged.",
            "sideEffectsReview": "Minor headaches in the first week, but they subsided.",
            "commentsReview": "Highly recommend discussing with your doctor. Life-changing for me.",
        }
        st.rerun()
    
    if st.button("👎 Negative Review Example"):
        st.session_state["example"] = {
            "urlDrugName": "Ambien",
            "condition": "Insomnia",
            "rating": 3.0,
            "sideEffects": "Severe Side Effects",
            "benefitsReview": "Did help me fall asleep initially.",
            "sideEffectsReview": "Terrible next-day grogginess, memory issues, and strange behavior.",
            "commentsReview": "Had to stop taking it due to side effects. Not worth it for me.",
        }
        st.rerun()
    
    if st.button("😐 Neutral Review Example"):
        st.session_state["example"] = {
            "urlDrugName": "Metformin",
            "condition": "Type 2 Diabetes",
            "rating": 5.0,
            "sideEffects": "Moderate Side Effects",
            "benefitsReview": "Blood sugar levels improved somewhat.",
            "sideEffectsReview": "Stomach upset and nausea, especially early on.",
            "commentsReview": "Works okay but not amazing. Still on the fence about continuing.",
        }
        st.rerun()

    st.markdown("---")
    st.header("ℹ️ About")
    st.write(
        "This app uses machine learning to predict drug effectiveness "
        "based on patient reviews. The model was trained on drug reviews "
        "from DrugLib.com."
    )
    st.write(
        "**Effectiveness Classes:**\n"
        "- Ineffective\n"
        "- Marginally Effective\n"
        "- Moderately Effective\n"
        "- Considerably Effective\n"
        "- Highly Effective"
    )
