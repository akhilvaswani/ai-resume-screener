"""
AI Resume Screener - Core Screening Engine
Matches resumes against job descriptions using embedding similarity
and structured skill extraction.
"""

import re
from embeddings import EmbeddingManager
from skill_extractor import SkillExtractor
from config import CONFIG


class ResumeScreener:
    """
    Screens resumes against job descriptions using a combination of
    semantic embedding similarity and structured skill matching.
    """

    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.skill_extractor = SkillExtractor()
        self.semantic_weight = CONFIG["SEMANTIC_WEIGHT"]
        self.skill_weight = CONFIG["SKILL_WEIGHT"]

    def screen(self, resume_text, job_description_text):
        """
        Screen a resume against a job description.

        Args:
            resume_text: The full text of the resume
            job_description_text: The full text of the job description

        Returns:
            dict with overall score, breakdown, matched/missing skills,
            and recommendation
        """
        # Clean inputs
        resume_clean = self._clean_text(resume_text)
        job_clean = self._clean_text(job_description_text)

        if not resume_clean or not job_clean:
            return self._empty_result("Empty resume or job description")

        # Extract skills from both documents
        resume_skills = self.skill_extractor.extract(resume_clean)
        job_skills = self.skill_extractor.extract(job_clean)

        # Calculate semantic similarity
        semantic_score = self._calculate_semantic_similarity(
            resume_clean, job_clean
        )

        # Calculate category-level semantic scores
        category_scores = self._calculate_category_scores(
            resume_skills, job_skills
        )

        # Calculate skill overlap
        skill_result = self._calculate_skill_overlap(resume_skills, job_skills)

        # Compute weighted overall score
        overall_score = (
            self.semantic_weight * semantic_score +
            self.skill_weight * skill_result["skill_score"]
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_score, semantic_score, skill_result
        )

        return {
            "overall_score": round(overall_score, 4),
            "semantic_score": round(semantic_score, 4),
            "skill_score": round(skill_result["skill_score"], 4),
            "category_scores": category_scores,
            "matched_skills": skill_result["matched"],
            "missing_skills": skill_result["missing"],
            "bonus_skills": skill_result["bonus"],
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "recommendation": recommendation,
        }

    def _clean_text(self, text):
        """Clean and normalize text for processing."""
        if not text:
            return ""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep common punctuation
        text = re.sub(r"[^\w\s.,;:!?@#$%&*()\-/'+]", "", text)
        return text.strip()

    def _calculate_semantic_similarity(self, resume_text, job_text):
        """Calculate cosine similarity between full document embeddings."""
        resume_embedding = self.embedding_manager.get_embedding(resume_text)
        job_embedding = self.embedding_manager.get_embedding(job_text)

        if resume_embedding is None or job_embedding is None:
            return 0.0

        return self._cosine_similarity(resume_embedding, job_embedding)

    def _calculate_category_scores(self, resume_skills, job_skills):
        """Calculate semantic similarity for each skill category."""
        categories = set(list(resume_skills.keys()) + list(job_skills.keys()))
        scores = {}

        for category in categories:
            resume_cat = resume_skills.get(category, [])
            job_cat = job_skills.get(category, [])

            if not job_cat:
                continue

            if not resume_cat:
                scores[category] = 0.0
                continue

            resume_text = ", ".join(resume_cat)
            job_text = ", ".join(job_cat)

            resume_emb = self.embedding_manager.get_embedding(resume_text)
            job_emb = self.embedding_manager.get_embedding(job_text)

            if resume_emb is not None and job_emb is not None:
                scores[category] = round(
                    self._cosine_similarity(resume_emb, job_emb), 4
                )
            else:
                scores[category] = 0.0

        return scores

    def _calculate_skill_overlap(self, resume_skills, job_skills):
        """Calculate what percentage of required skills are covered."""
        # Flatten all skills to sets for comparison
        resume_all = set()
        for skills in resume_skills.values():
            resume_all.update(s.lower() for s in skills)

        job_all = set()
        for skills in job_skills.values():
            job_all.update(s.lower() for s in skills)

        if not job_all:
            return {
                "skill_score": 0.0,
                "matched": [],
                "missing": [],
                "bonus": list(resume_all)
            }

        matched = resume_all & job_all
        missing = job_all - resume_all
        bonus = resume_all - job_all

        # Also check for fuzzy matches (e.g., "javascript" vs "js")
        fuzzy_matched = set()
        for job_skill in missing.copy():
            for resume_skill in bonus.copy():
                if self._is_fuzzy_match(job_skill, resume_skill):
                    fuzzy_matched.add(job_skill)
                    matched.add(f"{resume_skill} (matched: {job_skill})")
                    missing.discard(job_skill)
                    break

        skill_score = len(matched) / len(job_all) if job_all else 0.0

        return {
            "skill_score": min(skill_score, 1.0),
            "matched": sorted(matched),
            "missing": sorted(missing),
            "bonus": sorted(bonus - fuzzy_matched)
        }

    def _is_fuzzy_match(self, skill1, skill2):
        """Check if two skills are fuzzy matches (abbreviations, etc.)."""
        abbreviations = {
            "javascript": ["js", "es6", "es2015"],
            "typescript": ["ts"],
            "python": ["py"],
            "kubernetes": ["k8s"],
            "amazon web services": ["aws"],
            "google cloud platform": ["gcp"],
            "microsoft azure": ["azure"],
            "postgresql": ["postgres"],
            "mongodb": ["mongo"],
            "machine learning": ["ml"],
            "artificial intelligence": ["ai"],
            "continuous integration": ["ci"],
            "continuous deployment": ["cd"],
            "ci/cd": ["cicd", "ci cd"],
        }

        for full_name, abbrevs in abbreviations.items():
            names = [full_name] + abbrevs
            if skill1 in names and skill2 in names:
                return True

        # Check if one contains the other
        if len(skill1) > 3 and len(skill2) > 3:
            if skill1 in skill2 or skill2 in skill1:
                return True

        return False

    def _cosine_similarity(self, vec_a, vec_b):
        """Calculate cosine similarity between two vectors."""
        import numpy as np

        a = np.array(vec_a)
        b = np.array(vec_b)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _generate_recommendation(self, overall_score, semantic_score, skill_result):
        """Generate a human-readable recommendation."""
        if overall_score >= CONFIG["SIMILARITY_THRESHOLD"]:
            level = "Strong Match"
        elif overall_score >= CONFIG["POTENTIAL_THRESHOLD"]:
            level = "Potential Match"
        else:
            level = "Not Recommended"

        matched_count = len(skill_result["matched"])
        missing_count = len(skill_result["missing"])
        total_required = matched_count + missing_count

        reasons = []

        if semantic_score >= 0.8:
            reasons.append("Resume content closely aligns with job requirements")
        elif semantic_score >= 0.6:
            reasons.append("Resume shows moderate alignment with role")
        else:
            reasons.append("Resume content does not closely match this role")

        if total_required > 0:
            coverage = matched_count / total_required
            if coverage >= 0.8:
                reasons.append(
                    f"Strong skill coverage ({matched_count}/{total_required} required skills)"
                )
            elif coverage >= 0.5:
                reasons.append(
                    f"Partial skill coverage ({matched_count}/{total_required} required skills)"
                )
            else:
                reasons.append(
                    f"Low skill coverage ({matched_count}/{total_required} required skills)"
                )

        if skill_result["bonus"]:
            bonus_count = len(skill_result["bonus"])
            reasons.append(f"Has {bonus_count} additional relevant skills")

        if skill_result["missing"]:
            top_missing = skill_result["missing"][:5]
            reasons.append(f"Key missing skills: {', '.join(top_missing)}")

        return {
            "level": level,
            "score": round(overall_score * 100, 1),
            "reasons": reasons
        }

    def _empty_result(self, reason):
        """Return an empty result with an error message."""
        return {
            "overall_score": 0.0,
            "semantic_score": 0.0,
            "skill_score": 0.0,
            "category_scores": {},
            "matched_skills": [],
            "missing_skills": [],
            "bonus_skills": [],
            "resume_skills": {},
            "job_skills": {},
            "recommendation": {
                "level": "Error",
                "score": 0,
                "reasons": [reason]
            }
        }
