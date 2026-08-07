"""
SmartHire Streamlit App
Classifies a resume, recommends matching jobs, shows skill-gap chips,
an aggregated missing-skills summary, and a category confidence chart.
Run from project root with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import joblib
import pdfplumber
import docx
from collections import Counter

from src import config
from src.features.text_features import clean_text
from src.features.match_features import get_job_skills, skill_gap
from src.models.recommender import recommend_jobs


# ---------- Page config ----------
st.set_page_config(page_title="SmartHire", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0;
        color: #1a3a5c;
    }
    .subtitle {color: #3d4f61; font-size: 1.3rem; margin-top: 0;}

    .stButton>button {
        width: 100%;
        padding: 0.7rem;
        font-weight: 600;
        font-size: 1.05rem;
        border-radius: 8px;
    }

    div[data-testid="stExpander"] p, div[data-testid="stExpander"] summary {
        font-size: 1.1rem;
    }

    h1, h2, h3 {
        font-size: 1.9rem !important;
    }

    .skill-chip {
        display: inline-block; padding: 6px 14px; margin: 5px 7px 5px 0;
        border-radius: 16px; font-size: 1rem; font-weight: 500;
    }
    .chip-matched {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .chip-missing {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    .chip-count {background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;}
    </style>
""", unsafe_allow_html=True)


# ---------- Load models once ----------
@st.cache_resource
def load_models():
    clf = joblib.load(config.MODELS_DIR / "resume_classifier.pkl")
    vectorizer = joblib.load(config.MODELS_DIR / "tfidf_vectorizer.pkl")
    return clf, vectorizer

clf, vectorizer = load_models()


# ---------- Load sample resumes for the demo buttons ----------
@st.cache_data
def load_sample_resumes():
    df = pd.read_csv(config.RESUMES_CLEAN_FILE)
    sample_categories = ["HR", "INFORMATION-TECHNOLOGY", "SALES", "TEACHER", "FINANCE", "ENGINEERING"]
    samples = {}
    for cat in sample_categories:
        subset = df[df["Category"] == cat]
        if not subset.empty:
            samples[cat] = subset.iloc[0]["Resume_str"]
    return samples

sample_resumes = load_sample_resumes()


# ---------- File extraction helpers ----------
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

def render_skill_chips(skills, css_class):
    if not skills:
        return "<p style='color:#999;'>None</p>"
    return "".join(f'<span class="skill-chip {css_class}">{s}</span>' for s in skills)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📋 How it works")
    st.markdown("""
    1. Try a sample resume, paste your own, or upload a file
    2. Click **Analyze Resume**
    3. Get your predicted category + confidence breakdown
    4. See top matching jobs, skill gaps, and what to learn next
    """)
    st.divider()
    st.markdown("### ⚙️ About")
    st.caption(
        "Category prediction: Logistic Regression on TF-IDF. "
        "Job matching: TF-IDF + cosine similarity. "
        "Skill gap: keyword overlap between resume and job requirements. "
        "Fully offline — no external API calls."
    )
    top_n = st.slider("Number of job matches to show", min_value=3, max_value=10, value=5)


# ---------- Header ----------
st.markdown('<p class="main-header">🎯 SmartHire</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Resume Classifier, Job Matcher & Skill-Gap Report</p>', unsafe_allow_html=True)
st.write("")

# ---------- Sample resume quick-try ----------
st.markdown("**🚀 Try an example instantly:**")
sample_cols = st.columns(len(sample_resumes))
selected_sample = None
for i, (cat, text) in enumerate(sample_resumes.items()):
    if sample_cols[i].button(f"📄 {cat.title()}"):
        selected_sample = text

st.write("")

# ---------- Input ----------
input_method = st.radio("Or choose input method:", ["📋 Paste text", "📁 Upload file"], horizontal=True)
resume_text = selected_sample if selected_sample else ""

if input_method == "📋 Paste text":
    resume_text = st.text_area("Paste your resume text here:", value=resume_text, height=250, placeholder="Paste your full resume text...")
else:
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file is not None:
        with st.spinner("Extracting text..."):
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
        if not resume_text:
            st.error("Couldn't extract text — this may be a scanned/image PDF. Try pasting the text instead.")
        else:
            with st.expander("📄 Extracted text (click to review/edit)", expanded=False):
                resume_text = st.text_area("Edit if needed:", value=resume_text, height=250, label_visibility="collapsed")


# ---------- Analyze ----------
st.write("")
if st.button("🔍 Analyze Resume", type="primary"):
    if not resume_text or not resume_text.strip():
        st.warning("Please try a sample, paste, or upload a resume first.")
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

        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.subheader("📊 Predicted Category")
            st.metric(label="Best match", value=predicted_category, delta=f"{confidence:.1f}% confidence")
            if confidence < 25:
                st.caption("⚠️ Low confidence — resume text may be short or use unusual phrasing for this model.")
            st.caption("ℹ️ This category is broad — check the job matches below for more specific role alignment.")

        with col2:
            st.subheader("📈 Confidence Across All Categories")
            prob_df = pd.DataFrame({
                "Category": clf.classes_,
                "Confidence": probabilities * 100
            }).sort_values("Confidence", ascending=True).tail(10)
            st.bar_chart(prob_df.set_index("Category"))

        st.divider()

        # ---------- Job matches + skill gap per job ----------
        st.subheader(f"💼 Top {top_n} Matching Jobs & Skill Gaps")

        all_missing_skills = []

        if recommendations.empty:
            st.info("No matching jobs found.")
        else:
            for _, job in recommendations.iterrows():
                score_pct = round(job["score"] * 100, 1)
                job_title = str(job["Job Title"]).strip()
                with st.expander(f"{job_title} — {score_pct}% match"):
                    job_skills = get_job_skills(job)
                    matched, missing = skill_gap(resume_text, job_skills)
                    all_missing_skills.extend(missing)

                    sub1, sub2 = st.columns(2)
                    with sub1:
                        st.markdown(f"**✅ Matched Skills ({len(matched)})**")
                        st.markdown(render_skill_chips(matched, "chip-matched"), unsafe_allow_html=True)
                    with sub2:
                        st.markdown(f"**❌ Missing Skills ({len(missing)})**")
                        st.markdown(render_skill_chips(missing, "chip-missing"), unsafe_allow_html=True)

            csv = recommendations.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download job matches as CSV", csv, "job_matches.csv", "text/csv")

            st.divider()

            # ---------- Aggregated skill-gap summary ----------
            st.subheader("🎯 What to Learn Next")
            st.caption("Skills most frequently missing across your top job matches — ranked by how often they appear.")

            if all_missing_skills:
                skill_counts = Counter(s.lower() for s in all_missing_skills)
                top_gaps = skill_counts.most_common(10)
                gap_html = "".join(
                    f'<span class="skill-chip chip-count">{skill} ({count}x)</span>'
                    for skill, count in top_gaps
                )
                st.markdown(gap_html, unsafe_allow_html=True)
            else:
                st.write("No common skill gaps found — your resume covers most requirements across these matches! 🎉")