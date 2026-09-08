"""
Placement Readiness Intelligence System (PRIS)
Streamlit app: upload a resume PDF + paste a job description -> Gemini/Groq extract
structured skills/projects/etc -> numeric features are built from that structured data
-> the trained ML model (readiness_model.pkl, in this same folder) predicts a
readiness label + score.

Run:
    streamlit run app.py

API key:
    Put GEMINI_API_KEY and/or GROQ_API_KEY in a .env file next to this script
    (see .env.example), OR paste a key directly in the sidebar at runtime.
"""

import io
import json
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

MODEL_DIR = os.path.dirname(__file__)  # readiness_model.pkl etc. live at repo root, next to app.py

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="PRIS - Placement Readiness Intelligence System", page_icon="🎯", layout="wide")


# ----------------------------------------------------------------------------
# Cached artifact loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "readiness_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))
    return model, scaler, label_encoder, metadata


try:
    MODEL, SCALER, LABEL_ENCODER, METADATA = load_artifacts()
    FEATURE_COLS = METADATA["feature_columns"]
    ARTIFACTS_OK = True
except Exception as e:
    ARTIFACTS_OK = False
    ARTIFACT_ERROR = str(e)


# ----------------------------------------------------------------------------
# Sidebar: API key + provider
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("### LLM Provider")
provider = st.sidebar.selectbox("Provider", ["Groq", "Gemini"], index=0)

env_key_name = "GEMINI_API_KEY" if provider == "Gemini" else "GROQ_API_KEY"
env_key_value = os.getenv(env_key_name, "")

use_env_key = st.sidebar.checkbox(
    f"Use {env_key_name} from .env",
    value=bool(env_key_value),
    help="Untick to paste a key manually instead of reading .env",
)

if use_env_key and env_key_value:
    api_key = env_key_value
    st.sidebar.success(f"Loaded {env_key_name} from .env")
else:
    api_key = st.sidebar.text_input(
        f"{provider} API key",
        type="password",
        placeholder="Paste your API key here",
    )

gemini_model_name = st.sidebar.text_input("Gemini model", value="gemini-1.5-flash") if provider == "Gemini" else None
groq_model_name = st.sidebar.text_input("Groq model", value="openai/gpt-oss-120b") if provider == "Groq" else None

st.sidebar.markdown("---")
st.sidebar.caption(
    "Key is only used in-memory for this session's API calls. "
    "Put it in a `.env` file (see `.env.example`) to avoid re-entering it."
)


# ----------------------------------------------------------------------------
# LLM calls
# ----------------------------------------------------------------------------
RESUME_PROMPT = """You are a resume parser. Read the resume text below and return ONLY a
single valid JSON object (no markdown fences, no commentary) with this exact schema:

{{
  "skills": ["list", "of", "technical/tool skills mentioned"],
  "soft_skills": ["list", "of", "soft skills mentioned"],
  "projects_count": <integer>,
  "certifications_count": <integer>,
  "internships_count": <integer>,
  "has_education_section": <true/false>,
  "resume_category": "<one short phrase, e.g. 'Software/CSE', 'Electronics', 'Mechanical'>"
}}

Resume text:
---
{text}
---
"""

JD_PROMPT = """You are a job-description parser. Read the job description text below and
return ONLY a single valid JSON object (no markdown fences, no commentary) with this exact
schema:

{{
  "job_title": "<string>",
  "role_category": "<one short phrase, e.g. 'Software Engineering', 'Data Science'>",
  "required_skills": ["list", "of", "required technical skills/tools"],
  "critical_skills": ["subset of required_skills that are must-have / most critical"],
  "optional_skills": ["list", "of", "nice-to-have skills"],
  "soft_skills": ["list", "of", "soft skills mentioned"]
}}

Job description text:
---
{text}
---
"""


def _extract_json(raw_text: str) -> dict:
    """Strip markdown code fences etc. and parse the first JSON object found."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw_text[:300]}")
    return json.loads(match.group(0))


def call_gemini(prompt: str, api_key: str, model_name: str) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return _extract_json(response.text)


def call_groq(prompt: str, api_key: str, model_name: str) -> dict:
    from groq import Groq

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You return only valid JSON, no commentary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return _extract_json(completion.choices[0].message.content)


def call_llm(prompt: str) -> dict:
    if not api_key:
        raise RuntimeError(f"No {provider} API key set. Add it in the sidebar or in .env.")
    if provider == "Gemini":
        return call_gemini(prompt, api_key, gemini_model_name)
    return call_groq(prompt, api_key, groq_model_name)


def call_llm_feedback(resume_json: dict, jd_json: dict, matched, missing, critical_missing, score, label) -> str:
    """Ask the LLM for the human-readable summary + improvement plan."""
    prompt = f"""You are a placement readiness coach. Given this analysis, write:
1) A 10-20 line plain-English summary of whether the candidate suits this role.
2) A short skill gap report (what to learn first).
3) A 7-day improvement plan (bullet points).
4) A 30-day improvement plan (bullet points).

Candidate skills: {resume_json.get('skills', [])}
Job title: {jd_json.get('job_title', 'N/A')}
Matched skills: {matched}
Missing skills: {missing}
Critical missing skills: {critical_missing}
Placement Readiness Score: {score}/100
Readiness Level: {label}

Return plain text (markdown headings allowed), no JSON."""
    if not api_key:
        return "_(No API key set - add one in the sidebar to generate personalized feedback.)_"
    try:
        if provider == "Gemini":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(gemini_model_name)
            return model.generate_content(prompt).text
        else:
            from groq import Groq

            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=groq_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            return completion.choices[0].message.content
    except Exception as e:
        return f"_(Feedback generation failed: {e})_"


# ----------------------------------------------------------------------------
# Feature Builder — mirrors the feature engineering used to train the model
# (see 01_data_preprocessing_eda.ipynb / 02_model_training.ipynb)
# ----------------------------------------------------------------------------
TIER_KEYWORDS = {
    "tier-1": 100, "tier1": 100,
    "tier-2": 70, "tier2": 70,
    "tier-3": 45, "tier3": 45,
}


def build_features(resume_json: dict, jd_json: dict, full_resume_text: str, full_jd_text: str):
    resume_skills = {s.strip().lower() for s in resume_json.get("skills", []) if s.strip()}
    required_skills = {s.strip().lower() for s in jd_json.get("required_skills", []) if s.strip()}
    critical_skills = {s.strip().lower() for s in jd_json.get("critical_skills", []) if s.strip()}
    optional_skills = {s.strip().lower() for s in jd_json.get("optional_skills", []) if s.strip()}

    matched = sorted(resume_skills & required_skills)
    missing = sorted(required_skills - resume_skills)
    critical_matched = sorted(resume_skills & critical_skills)
    critical_missing = sorted(critical_skills - resume_skills)

    skill_match_percentage = (len(matched) / len(required_skills) * 100) if required_skills else 50.0
    critical_skill_match_percentage = (
        (len(critical_matched) / len(critical_skills) * 100) if critical_skills else skill_match_percentage
    )
    missing_skills_count = len(missing)
    critical_missing_skills_count = len(critical_missing)

    projects_count = int(resume_json.get("projects_count", 0) or 0)
    certifications_count = int(resume_json.get("certifications_count", 0) or 0)
    internships_count = int(resume_json.get("internships_count", 0) or 0)

    project_relevance_score = min(projects_count / 5 * 100, 100)
    certification_relevance_score = min(certifications_count / 4 * 100, 100)
    internship_relevance_score = min(internships_count / 3 * 100, 100)

    has_education = bool(resume_json.get("has_education_section", False))
    resume_completeness_score = (
        (projects_count > 0)
        + (certifications_count > 0)
        + (internships_count > 0)
        + (len(resume_skills) > 0)
        + has_education
        + (len(resume_json.get("soft_skills", [])) > 0)
    ) / 6 * 100

    # keyword overlap between full resume text and all JD keywords (skills + soft skills)
    jd_keywords = required_skills | optional_skills | {
        s.strip().lower() for s in jd_json.get("soft_skills", []) if s.strip()
    }
    resume_text_lower = full_resume_text.lower()
    if jd_keywords:
        hits = sum(1 for kw in jd_keywords if kw and kw in resume_text_lower)
        keyword_match_score = hits / len(jd_keywords) * 100
    else:
        keyword_match_score = 50.0

    resume_category = str(resume_json.get("resume_category", "")).lower()
    role_category = str(jd_json.get("role_category", "")).lower()
    role_category_match_score = 100.0 if resume_category and resume_category in role_category or role_category in resume_category else 55.0

    features = {
        "skill_match_percentage": skill_match_percentage,
        "critical_skill_match_percentage": critical_skill_match_percentage,
        "missing_skills_count": missing_skills_count,
        "critical_missing_skills_count": critical_missing_skills_count,
        "project_relevance_score": project_relevance_score,
        "certification_relevance_score": certification_relevance_score,
        "internship_relevance_score": internship_relevance_score,
        "resume_completeness_score": resume_completeness_score,
        "keyword_match_score": keyword_match_score,
        "role_category_match_score": role_category_match_score,
    }
    report = {
        "matched": matched,
        "missing": missing,
        "critical_matched": critical_matched,
        "critical_missing": critical_missing,
    }
    return features, report


def predict_readiness(features: dict):
    X = np.array([[features[c] for c in FEATURE_COLS]])
    if METADATA.get("uses_scaler"):
        X = SCALER.transform(X)
    pred_idx = MODEL.predict(X)[0]
    proba = MODEL.predict_proba(X)[0]
    label = LABEL_ENCODER.inverse_transform([pred_idx])[0]

    midpoints = METADATA["class_score_midpoint"]
    classes = LABEL_ENCODER.classes_
    score = sum(midpoints[c] * p for c, p in zip(classes, proba))
    confidence = float(np.max(proba)) * 100
    return label, round(float(score), 1), round(confidence, 1), dict(zip(classes, proba))


def _sanitize_pdf_text(text: str) -> str:
    """Normalize LLM output for reportlab's base fonts. Smart punctuation -> ASCII,
    then drop anything the base Helvetica font can't render (emoji, keycap digits,
    CJK, etc.) instead of letting it show up as a black tofu box."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "-",
        "\ufe0f": "", "\u20e3": "",  # emoji variation selector / combining keycap
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def _clean_for_paragraph(text: str) -> str:
    """Sanitize + XML-escape + convert **bold**/*italic* markdown to reportlab tags,
    for text that goes into a Paragraph (which parses a small XML markup subset)."""
    import html
    text = _sanitize_pdf_text(text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def build_pdf_report(jd_json, score, label, confidence, features, skill_report, feedback_text) -> bytes:
    """Render the analysis as a formatted PDF (tables + sections) using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1f2933"))
    body = styles["Normal"]
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=9, textColor=colors.grey)

    story = [
        Paragraph("Placement Readiness Report", h1),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", small),
        Spacer(1, 10),
        HRFlowable(width="100%", color=colors.HexColor("#cbd2d9")),
        Spacer(1, 10),
    ]

    # --- Summary table ---
    story.append(Paragraph("Summary", h2))
    summary_data = [
        ["Job Title", _sanitize_pdf_text(jd_json.get("job_title", "N/A"))],
        ["Role Category", _sanitize_pdf_text(jd_json.get("role_category", "N/A"))],
        ["Placement Readiness Score", f"{score:.0f} / 100"],
        ["Readiness Level", label],
        ["Model Confidence", f"{confidence:.0f}%"],
    ]
    summary_tbl = Table(summary_data, colWidths=[2.2 * inch, 3.8 * inch])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd2d9")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_tbl)

    # --- Skill gap table ---
    story.append(Paragraph("Skill Gap Analysis", h2))
    rows = [["Skill", "Status", "Critical"]]
    for s in skill_report["matched"]:
        s = _sanitize_pdf_text(s)
        rows.append([s, "Matched", "Yes" if s in [_sanitize_pdf_text(x) for x in skill_report["critical_matched"]] else "No"])
    for s in skill_report["missing"]:
        s = _sanitize_pdf_text(s)
        rows.append([s, "Missing", "Yes" if s in [_sanitize_pdf_text(x) for x in skill_report["critical_missing"]] else "No"])
    if len(rows) == 1:
        rows.append(["-", "-", "-"])
    skill_tbl = Table(rows, colWidths=[2.6 * inch, 1.7 * inch, 1.7 * inch], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334e68")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd2d9")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i, r in enumerate(rows[1:], start=1):
        if r[1] == "Matched":
            style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#0f7b0f")))
        elif r[1] == "Missing":
            style_cmds.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#c0392b")))
    skill_tbl.setStyle(TableStyle(style_cmds))
    story.append(skill_tbl)

    # --- Feature scores table ---
    story.append(Paragraph("Feature Scores", h2))
    feat_rows = [["Feature", "Value"]] + [
        [k.replace("_", " ").title(), f"{v:.1f}"] for k, v in features.items()
    ]
    feat_tbl = Table(feat_rows, colWidths=[3.5 * inch, 2.5 * inch], repeatRows=1)
    feat_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334e68")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd2d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(feat_tbl)

    # --- AI feedback (plain text, headings on lines starting with #) ---
    story.append(Paragraph("AI Feedback & Improvement Plan", h2))
    for raw_line in (feedback_text or "").split("\n"):
        line = _sanitize_pdf_text(raw_line.strip())
        if not line:
            story.append(Spacer(1, 4))
            continue
        if line.startswith("#"):
            text = _clean_for_paragraph(line.lstrip("#").strip())
            story.append(Paragraph(text, styles["Heading3"]))
        elif line.startswith(("-", "*")):
            text = _clean_for_paragraph(line.lstrip("-*").strip())
            story.append(Paragraph("- " + text, body))
        else:
            # numbered headings like "1. Summary" (incl. emoji-numeral bullets, already
            # stripped by _sanitize_pdf_text above) -> promote to a small heading
            m = re.match(r"^(\d{1,2})[.):]?\s+(.+)$", line)
            heading_text = m.group(2) if m else None
            heading_plain = re.sub(r"\*+", "", heading_text) if heading_text else ""
            if m and len(heading_plain) < 90:
                story.append(Paragraph(_clean_for_paragraph(heading_text), styles["Heading3"]))
            else:
                story.append(Paragraph(_clean_for_paragraph(line), body))

    doc.build(story)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🎯 Placement Readiness Intelligence System")
st.caption("Upload a resume + paste a job description to get an ML-based placement readiness score.")

if not ARTIFACTS_OK:
    st.error(
        "Could not load model artifacts from the app folder "
        f"({ARTIFACT_ERROR}). Run 02_model_training.ipynb first to generate "
        "readiness_model.pkl, scaler.pkl, label_encoder.pkl, metadata.pkl "
        "in the same folder as app.py."
    )
    st.stop()

if "analysis" not in st.session_state:
    st.session_state["analysis"] = None

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Student Resume (PDF)", type=["pdf"])
with col2:
    jd_text_input = st.text_area("Job Description", height=250, placeholder="Paste the job description here...")

analyze_clicked = st.button("🔍 Analyze Readiness", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Run analysis ONCE per click and stash everything in session_state.
# Streamlit reruns this whole script on every widget interaction (including
# clicking a download button), so without session_state the results block
# below would vanish the moment someone clicks "Download PDF".
# ----------------------------------------------------------------------------
if analyze_clicked:
    if resume_file is None:
        st.warning("Upload a resume PDF first.")
        st.stop()
    if not jd_text_input.strip():
        st.warning("Paste a job description first.")
        st.stop()
    if not api_key:
        st.warning(f"No {provider} API key set. Add one in the sidebar (or .env) to run the analysis.")
        st.stop()

    with st.spinner("Extracting resume text..."):
        resume_text = extract_pdf_text(resume_file)
    if not resume_text.strip():
        st.error("Couldn't extract any text from that PDF (it may be a scanned image).")
        st.stop()

    with st.spinner(f"Analyzing resume with {provider}..."):
        try:
            resume_json = call_llm(RESUME_PROMPT.format(text=resume_text[:12000]))
        except Exception as e:
            st.error(f"Resume analysis failed: {e}")
            st.stop()

    with st.spinner(f"Analyzing job description with {provider}..."):
        try:
            jd_json = call_llm(JD_PROMPT.format(text=jd_text_input[:8000]))
        except Exception as e:
            st.error(f"Job description analysis failed: {e}")
            st.stop()

    features, skill_report = build_features(resume_json, jd_json, resume_text, jd_text_input)
    label, score, confidence, proba_map = predict_readiness(features)

    with st.spinner("Generating personalized feedback..."):
        feedback_text = call_llm_feedback(
            resume_json, jd_json,
            skill_report["matched"], skill_report["missing"], skill_report["critical_missing"],
            score, label,
        )

    st.session_state["analysis"] = dict(
        resume_json=resume_json, jd_json=jd_json, features=features,
        skill_report=skill_report, label=label, score=score,
        confidence=confidence, proba_map=proba_map, feedback_text=feedback_text,
    )

# ----------------------------------------------------------------------------
# Render from session_state (survives reruns triggered by download buttons etc.)
# ----------------------------------------------------------------------------
result = st.session_state["analysis"]

if result is None:
    st.info("Upload a resume PDF and paste a job description, then click **Analyze Readiness**.")
else:
    resume_json = result["resume_json"]
    jd_json = result["jd_json"]
    features = result["features"]
    skill_report = result["skill_report"]
    label = result["label"]
    score = result["score"]
    confidence = result["confidence"]
    proba_map = result["proba_map"]
    feedback_text = result["feedback_text"]

    st.success("Analysis complete.")
    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Placement Readiness Score", f"{score:.0f} / 100")
    m2.metric("Readiness Level", label)
    m3.metric("Model Confidence", f"{confidence:.0f}%")

    st.progress(min(max(score / 100, 0.0), 1.0))

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Match Report", "📊 Features & Probabilities", "💡 AI Feedback", "⬇️ Download"])

    with tab1:
        st.subheader(f"Job Title: {jd_json.get('job_title', 'N/A')}")
        st.caption(f"Role category: {jd_json.get('role_category', 'N/A')} | Resume category: {resume_json.get('resume_category', 'N/A')}")

        skill_rows = (
            [{"Skill": s, "Status": "✅ Matched", "Critical": "⭐ Yes" if s in skill_report["critical_matched"] else "No"} for s in skill_report["matched"]]
            + [{"Skill": s, "Status": "❌ Missing", "Critical": "🚨 Yes" if s in skill_report["critical_missing"] else "No"} for s in skill_report["missing"]]
        )
        if skill_rows:
            skill_df = pd.DataFrame(skill_rows).sort_values(["Status", "Critical"], ascending=[True, False])
            st.dataframe(skill_df, use_container_width=True, hide_index=True)
        else:
            st.info("No skills detected on either side to compare.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched", len(skill_report["matched"]))
        m2.metric("Missing", len(skill_report["missing"]))
        m3.metric("Critical matched", len(skill_report["critical_matched"]))
        m4.metric("Critical missing", len(skill_report["critical_missing"]))

    with tab2:
        feat_df = pd.DataFrame({"Feature": list(features.keys()), "Value": [round(v, 1) for v in features.values()]})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        proba_df = pd.DataFrame({"Readiness Level": list(proba_map.keys()), "Probability": [round(v * 100, 1) for v in proba_map.values()]})
        st.bar_chart(proba_df.set_index("Readiness Level"))

    with tab3:
        st.markdown(feedback_text)
        if st.button("🔄 Regenerate feedback"):
            with st.spinner("Generating personalized feedback..."):
                new_feedback = call_llm_feedback(
                    resume_json, jd_json,
                    skill_report["matched"], skill_report["missing"], skill_report["critical_missing"],
                    score, label,
                )
            st.session_state["analysis"]["feedback_text"] = new_feedback
            st.rerun()

    with tab4:
        report_lines = [
            f"# Placement Readiness Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"\n## Job: {jd_json.get('job_title', 'N/A')}",
            f"\n**Placement Readiness Score:** {score:.0f} / 100",
            f"**Readiness Level:** {label}",
            f"**Model Confidence:** {confidence:.0f}%",
            f"\n## Matched Skills\n" + (", ".join(skill_report["matched"]) or "None"),
            f"\n## Missing Skills\n" + (", ".join(skill_report["missing"]) or "None"),
            f"\n## Critical Missing Skills\n" + (", ".join(skill_report["critical_missing"]) or "None"),
            f"\n## AI Feedback\n{feedback_text}",
        ]
        report_md = "\n".join(report_lines)

        with st.spinner("Building PDF..."):
            pdf_bytes = build_pdf_report(jd_json, score, label, confidence, features, skill_report, feedback_text)

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "⬇️ Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"placement_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
                key="download_pdf",
            )
        with dcol2:
            st.download_button(
                "Download Report (Markdown)",
                data=report_md,
                file_name=f"placement_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_md",
            )