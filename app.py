import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Load the trained machine learning model
@st.cache_resource
def load_model():
    return joblib.load("models/spam_model.joblib")


# Main application
st.title("📧 Email Spam Detection System")

st.write(
    "This application uses Machine Learning "
    "to classify text as spam or legitimate."
)

st.divider()

# Check whether the model exists
model_path = Path("models/spam_model.joblib")

if not model_path.exists():
    st.error(
        "Trained model not found! "
        "Please run train_model.py first."
    )
    st.stop()

model = load_model()

# User input
message = st.text_area(
    "Enter your email or message below:",
    placeholder="Type or paste your message here...",
    height=180
)

# Prediction button
if st.button("🔍 Detect Spam", use_container_width=True):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        st.divider()
        st.subheader("Detection Result")

        if prediction == "spam":
            st.error("🚨 SPAM DETECTED")

            st.write(
                "The model identifies this text "
                "as resembling spam."
            )

        else:
            st.success("✅ LIKELY LEGITIMATE")

            st.write(
                "The model identifies this text "
                "as resembling a legitimate message."
            )

        st.caption(
            "This is an educational ML prediction, "
            "not a guarantee that a message is safe."
        )

st.divider()

st.caption("Built with Python, Scikit-learn and Streamlit")