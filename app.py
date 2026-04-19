import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

st.set_page_config(
    page_title="Mental Health Detection",
    page_icon="🧠",
    layout="centered"
)


# LOAD MODEL
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("best_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("best_model")
    return model, tokenizer

model, tokenizer = load_model()

# UI DESIGN
st.title("🧠 Mental Health Detection System")
st.markdown("Analyze text to detect signs of depression using AI.")

st.divider()

# Input box
user_input = st.text_area(
    "Enter text (e.g., tweets, messages, thoughts):",
    height=150,
    placeholder="Type something like 'I feel empty and tired all the time...'"
)

# PREDICTION FUNCTION
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)

    return predicted_class.item(), confidence.item(), probs[0][1].item()

# BUTTON
if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        pred, conf, dep_prob = predict(user_input)

        st.divider()

        # Result display
        if pred == 1:
            st.error("⚠️ Depression Detected")
        else:
            st.success("✅ No Depression Detected")

        # Confidence
        st.write(f"**Confidence:** {conf:.2%}")

        # Probability bar
        st.progress(dep_prob)

        st.caption(f"Depression Probability: {dep_prob:.2%}")
