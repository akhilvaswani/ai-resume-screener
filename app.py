"""
AI Resume Screener - Streamlit Web Interface
Interactive UI for screening resumes against job descriptions.
"""

import streamlit as st
import json
import os
import tempfile

from screener import ResumeScreener
from skill_extractor import SkillExtractor


@st.cache_resource
def load_screener():
    """Load the screener model (cached across reruns)."""
    return ResumeScreener()


def render_score_bar(score, label):
    """Render a visual score bar."""
    color = "#4CAF50" if score >= 0.7 else "#FF9800" if score >= 0.5 else "#F44336"
    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span>{label}</span>
            <span><strong>{score:.0%}</strong></span>
        </div>
        <div style="background-color: #e0e0e0; border-radius: 4px; height: 20px;">
            <div style="background-color: {color}; width: {score*100}%;
                        border-radius: 4px; height: 20px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI Resume Screener",
        page_icon="📋",
        layout="wide"
    )

    st.title("AI Resume Screener")
    st.markdown("Match resumes against job descriptions using AI-powered semantic analysis")

    # Sidebar configuration
    st.sidebar.header("Settings")
    semantic_weight = st.sidebar.slider(
        "Semantic Weight", 0.0, 1.0, 0.6, 0.05,
        help="Weight for overall text similarity"
    )
    skill_weight = 1.0 - semantic_weight
    st.sidebar.text(f"Skill Weight: {skill_weight:.2f}")

    threshold = st.sidebar.slider(
        "Match Threshold", 0.0, 1.0, 0.5, 0.05,
        help="Minimum score for 'Potential Match'"
    )

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Description")
        job_input_method = st.radio(
            "Input method:", ["Paste text", "Upload file"],
            key="job_input", horizontal=True
        )

        if job_input_method == "Paste text":
            job_text = st.text_area(
                "Paste the job description here:",
                height=300,
                placeholder="Enter the full job description..."
            )
        else:
            job_file = st.file_uploader(
                "Upload job description",
                type=["txt", "md"],
                key="job_file"
            )
            job_text = job_file.read().decode("utf-8") if job_file else ""

    with col2:
        st.subheader("Resume(s)")
        resume_input_method = st.radio(
            "Input method:", ["Paste text", "Upload file(s)"],
            key="resume_input", horizontal=True
        )

        resumes = []
        if resume_input_method == "Paste text":
            resume_text = st.text_area(
                "Paste the resume here:",
                height=300,
                placeholder="Enter the full resume text..."
            )
            if resume_text:
                resumes = [("Pasted Resume", resume_text)]
        else:
            resume_files = st.file_uploader(
                "Upload resume(s)",
                type=["txt", "md"],
                accept_multiple_files=True,
                key="resume_files"
            )
            for f in (resume_files or []):
                content = f.read().decode("utf-8")
                resumes.append((f.name, content))

    # Screen button
    if st.button("Screen Resumes", type="primary", use_container_width=True):
        if not job_text:
            st.error("Please provide a job description")
            return

        if not resumes:
            st.error("Please provide at least one resume")
            return

        screener = load_screener()

        # Update weights
        screener.semantic_weight = semantic_weight
        screener.skill_weight = skill_weight

        results = []
        progress = st.progress(0)

        for i, (name, resume_text) in enumerate(resumes):
            with st.spinner(f"Screening: {name}..."):
                result = screener.screen(resume_text, job_text)
                results.append((name, result))
            progress.progress((i + 1) / len(resumes))

        # Display results
        st.markdown("---")
        st.subheader("Results")

        # Sort by score
        results.sort(key=lambda x: x[1]["overall_score"], reverse=True)

        for rank, (name, result) in enumerate(results, 1):
            rec = result["recommendation"]
            rec_color = {
                "Strong Match": "green",
                "Potential Match": "orange",
                "Not Recommended": "red"
            }.get(rec["level"], "gray")

            with st.expander(
                f"#{rank} - {name} | Score: {result['overall_score']:.0%} | "
                f":{rec_color}[{rec['level']}]",
                expanded=(rank <= 3)
            ):
                # Score breakdown
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    render_score_bar(result["overall_score"], "Overall Score")
                with col_b:
                    render_score_bar(result["semantic_score"], "Semantic Match")
                with col_c:
                    render_score_bar(result["skill_score"], "Skill Coverage")

                # Skills
                sk_col1, sk_col2, sk_col3 = st.columns(3)
                with sk_col1:
                    st.markdown("**Matched Skills**")
                    for skill in result["matched_skills"]:
                        st.markdown(f"- {skill}")
                with sk_col2:
                    st.markdown("**Missing Skills**")
                    for skill in result["missing_skills"]:
                        st.markdown(f"- {skill}")
                with sk_col3:
                    st.markdown("**Bonus Skills**")
                    for skill in result["bonus_skills"][:10]:
                        st.markdown(f"- {skill}")

                # Recommendation
                st.markdown("**Reasoning:**")
                for reason in rec["reasons"]:
                    st.markdown(f"- {reason}")

        # Download results
        if results:
            results_json = json.dumps([
                {"candidate": name, **result}
                for name, result in results
            ], indent=2, default=str)

            st.download_button(
                "Download Results (JSON)",
                data=results_json,
                file_name="screening_results.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
