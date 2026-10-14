import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="📧",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

h1 {
    text-align: center;
    color: #60a5fa;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
}

.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1e293b;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/spam_model.joblib")


model_path = Path("models/spam_model.joblib")

st.title("📧 SpamGuard AI")
st.markdown(
    '<p class="subtitle">Machine Learning Text Spam Detector</p>',
    unsafe_allow_html=True
)

st.divider()

st.markdown("### 🔍 Analyse a Message")

st.write(
    "Paste a message below and let the machine learning "
    "model classify it."
)

if not model_path.exists():
    st.error("Model not found. Train the model first.")
    st.stop()

model = load_model()

message = st.text_area(
    "Message",
    placeholder="Paste your message here...",
    height=180
)

if st.button("🚀 Analyse Message"):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        st.divider()
        st.subheader("Analysis Result")

        if prediction == "spam":
            st.error("🚨 SPAM DETECTED")

            st.write(
                "The model classified this text as spam."
            )

        else:
            st.success("✅ LIKELY LEGITIMATE")

            st.write(
                "The model classified this text as legitimate."
            )

        st.caption(
            "Educational prototype: predictions can be incorrect. "
            "A legitimate-looking message is not necessarily safe."
        )

st.divider()

st.markdown(
    "<center>Built with Python • Scikit-learn • Streamlit</center>",
    unsafe_allow_html=True
)