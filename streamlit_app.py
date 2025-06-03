import streamlit as st
from tagger import generate_tags, generate_caption
from style_classifier import load_model, predict_style
from search import find_similar_images
from db_sqlite import init_db, save_metadata
import os
import json
from cloudinary_utils import upload_to_cloudinary
import os

# 📦 Init DB + Load style model
init_db()
with open("models/style_classes.json") as f:
    STYLE_CLASSES = json.load(f)
style_model = load_model("models/style_model_hf.pth", STYLE_CLASSES)

st.title("🎨 Curato - AI Art Tagger")

uploaded_file = st.file_uploader("Upload an artwork", type=["jpg", "png", "jpeg"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)  # ✅ Make sure 'data/' exists

    temp_path = os.path.join("data", uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())


    st.image(temp_path, caption="Uploaded Artwork", use_container_width=True)

    with st.spinner("Analyzing artwork..."):
        tags = generate_tags(temp_path)
        style = predict_style(temp_path, style_model, STYLE_CLASSES)
        caption = generate_caption(temp_path)

    st.success("Results:")
    st.write("🎨 **Art Style:**", style)
    st.write("🏷️ **Tags:**", tags)
    st.write("📝 **Caption:**", caption)
    cloud_url = upload_to_cloudinary(temp_path)
    st.write("🌐 **Cloudinary URL:**", cloud_url)


    # 💾 Save to SQLite
    save_metadata(uploaded_file.name, style, tags, caption, cloud_url)


    # 🔍 Image-to-Image Search
    st.markdown("---")
    st.subheader("🔍 Visually Similar Artworks")

    matches = find_similar_images(temp_path, top_k=5)

    if matches:
        cols = st.columns(len(matches))
        for col, (fname, score) in zip(cols, matches):
            img_path = os.path.join("gallery", fname)
            col.image(img_path, caption=f"{fname} (Score: {score:.2f})", use_container_width=True)
    else:
        st.info("No similar artworks found.")
