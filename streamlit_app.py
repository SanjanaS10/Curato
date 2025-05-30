import streamlit as st
from tagger import generate_tags
from style_classifier import load_model, predict_style
import os
import json

# Load class names dynamically
with open("models/style_classes.json") as f:
    STYLE_CLASSES = json.load(f)

style_model = load_model("models/style_model_hf.pth", STYLE_CLASSES)


st.title("🎨 Curato - AI Art Tagger")

uploaded_file = st.file_uploader("Upload an artwork", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # ✅ Save uploaded file first
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # ✅ Now it's safe to use temp_path
    st.image(temp_path, caption="Uploaded Artwork", use_column_width=True)

    with st.spinner("Analyzing artwork..."):
        tags = generate_tags(temp_path)
        style = predict_style(temp_path, style_model, STYLE_CLASSES)

    st.success("Results:")
    st.write("🎨 **Art Style:**", style)
    st.write("🏷️ **Tags:**", tags)
