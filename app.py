"""
Smart-Hire Streamlit App
Classifies a resume into a job category and recommends matching job postings.
Run with: streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import joblib
import pdfplumber
import docx

from src import config
from src.features.text_features import clean_text
from src.models.recommender import recommend_jobs


# ---------- Page config (must be first Streamlit command) ----------
st.set_page_config(
    page_title="Smart-Hire",
    page_icon="🎯",
    layout="wide",                     # use full browser width, not just centered narrow column
    initial_sidebar_state="expanded",
)

# ---------- Light custom styling ----------
st.markdown("""
    <style>
    .main-header {font-size: 2.2rem; font-weight: 700; margin-bottom: 0;}
    .subtitle {color: #666; font-size: 1rem; margin-top: 0;}
    .stButton>button {width: 100%; padding: 0.6rem; font-weight: 600;}
    </style>
""", unsafe_allow_html=True)


# ---------- Load models once (cached) ----------
@st.cache_resource
def load_models():
    clf = joblib.load(config.MODELS_DIR / "resume_classifier.pkl")
    vectorizer = joblib.load(config.MODELS_DIR / "tfidf_vectorizer.pkl")
    return clf, vectorizer

clf, vectorizer = load_models()


# ---------- File text extraction helpers ----------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs).strip()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📋 How it works")
    st.markdown("""
    1. Paste your resume text or upload a file
    2. Click **Analyze Resume**
    3. Get your predicted job category
    4. See the top 5 matching job postings
    """)
    st.divider()
    st.markdown("### ⚙️ About")
    st.caption(
        "Category prediction uses a Logistic Regression model trained on labeled resumes. "
        "Job matching uses TF-IDF + cosine similarity against real job postings. "
        "Runs fully offline — no external API calls."
    )
    top_n = st.slider("Number of job matches to show", min_value=3, max_value=10, value=5)


# ---------- Header ----------
st.markdown('<p class="main-header">🎯 Smart-Hire</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Resume Classifier & Job Matcher</p>', unsafe_allow_html=True)
st.write("")

# ---------- Input section ----------
input_method = st.radio("Choose input method:", ["📋 Paste text", "📁 Upload file"], horizontal=True)

resume_text = ""

if input_method == "📋 Paste text":
    resume_text = st.text_area("Paste your resume text here:", height=250, placeholder="Paste your full resume text...")

else:
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)

        if not resume_text:
            st.error("Couldn't extract text from this file — it may be a scanned/image-based PDF. Try pasting the text manually instead.")
        else:
            with st.expander("📄 Extracted text (click to review/edit)", expanded=False):
                resume_text = st.text_area("Edit if needed:", value=resume_text, height=250, label_visibility="collapsed")


# ---------- Analyze ----------
st.write("")
analyze_clicked = st.button("🔍 Analyze Resume", type="primary")

if analyze_clicked:
    if not resume_text or not resume_text.strip():
        st.warning("Please paste or upload a resume first.")
    else:
        with st.spinner("Analyzing resume..."):
            cleaned = clean_text(resume_text)
            vec = vectorizer.transform([cleaned])

            predicted_category = clf.predict(vec)[0]
            probabilities = clf.predict_proba(vec)[0]
            confidence = max(probabilities) * 100

            recommendations = recommend_jobs(resume_text, top_n=top_n)

        st.success("Analysis complete!")
        st.write("")

        # --- Results in two columns: category on left, top-3 alternatives on right ---
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Predicted Category")
            st.metric(label="Best match", value=predicted_category, delta=f"{confidence:.1f}% confidence")

            if confidence < 25:
                st.caption("⚠️ Low confidence — resume text may be short, unclear, or unusual for the categories this model was trained on.")

        with col2:
            st.subheader("🔝 Other Possible Categories")
            top3_idx = probabilities.argsort()[::-1][1:4]     # 2nd, 3rd, 4th best (skip the top one, already shown)
            for idx in top3_idx:
                cat = clf.classes_[idx]
                prob = probabilities[idx] * 100
                st.write(f"**{cat}** — {prob:.1f}%")

        st.divider()

        # --- Job recommendations table ---
        st.subheader(f"💼 Top {top_n} Matching Jobs")
        if recommendations.empty:
            st.info("No matching jobs found.")
        else:
            display_df = recommendations.copy()
            display_df["score"] = (display_df["score"] * 100).round(1).astype(str) + "%"
            display_df = display_df.rename(columns={"score": "Match %"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Let user download their results
            csv = recommendations.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download matches as CSV", csv, "job_matches.csv", "text/csv")