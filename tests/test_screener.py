"""
Tests for the AI Resume Screener components.
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skill_extractor import SkillExtractor


class TestSkillExtractor:
    """Tests for the skill extraction module."""

    def setup_method(self):
        self.extractor = SkillExtractor()

    def test_extract_programming_languages(self):
        text = "Experienced in Python, JavaScript, and Java development"
        skills = self.extractor.extract(text)
        assert "programming" in skills
        assert "Python" in skills["programming"]
        assert "JavaScript" in skills["programming"]
        assert "Java" in skills["programming"]

    def test_extract_cloud_skills(self):
        text = "Managed AWS infrastructure including EC2, S3, and Lambda"
        skills = self.extractor.extract(text)
        assert "cloud" in skills
        assert "AWS" in skills["cloud"]
        assert "EC2" in skills["cloud"]
        assert "S3" in skills["cloud"]
        assert "Lambda" in skills["cloud"]

    def test_extract_tools(self):
        text = "Built CI/CD pipelines with Docker and Kubernetes"
        skills = self.extractor.extract(text)
        assert "tools" in skills
        assert "Docker" in skills["tools"]
        assert "Kubernetes" in skills["tools"]

    def test_extract_frameworks(self):
        text = "Developed applications using React, Django, and Flask"
        skills = self.extractor.extract(text)
        assert "frameworks" in skills
        assert "React" in skills["frameworks"]
        assert "Django" in skills["frameworks"]
        assert "Flask" in skills["frameworks"]

    def test_extract_empty_text(self):
        skills = self.extractor.extract("")
        assert skills == {}

    def test_extract_no_skills(self):
        text = "I enjoy hiking and reading books on the weekend."
        skills = self.extractor.extract(text)
        # Should have very few or no technical skills
        total = sum(len(v) for v in skills.values())
        assert total <= 2  # Allow for minor false positives

    def test_extract_experience_years(self):
        text = "5+ years of experience in software development"
        skills = self.extractor.extract(text)
        assert "experience" in skills
        assert "5+ years" in skills["experience"]

    def test_case_insensitive(self):
        text = "Skills include python, JAVASCRIPT, and Docker"
        skills = self.extractor.extract(text)
        programming = skills.get("programming", [])
        assert "Python" in programming
        assert "JavaScript" in programming

    def test_short_skill_word_boundary(self):
        """Short skills like 'Go' or 'R' should match on word boundaries."""
        text = "Proficient in Go programming and R for statistics"
        skills = self.extractor.extract(text)
        programming = skills.get("programming", [])
        assert "Go" in programming
        assert "R" in programming

    def test_no_false_positive_short_skills(self):
        """Words like 'going' should not match 'Go'."""
        text = "I am going to the store to get some groceries"
        skills = self.extractor.extract(text)
        programming = skills.get("programming", [])
        assert "Go" not in programming

    def test_get_all_skills_flat(self):
        text = "Python, AWS, Docker, React developer"
        flat = self.extractor.get_all_skills_flat(text)
        assert isinstance(flat, list)
        assert len(flat) > 0
        assert all(isinstance(s, str) for s in flat)

    def test_compare_skills(self):
        resume = "Python, AWS, Docker, Kubernetes, Terraform"
        job = "Python, AWS, Kubernetes, Jenkins, Ansible"
        comparison = self.extractor.compare_skills(resume, job)
        assert "matched" in comparison
        assert "missing" in comparison
        assert "bonus" in comparison
        assert "coverage" in comparison
        assert 0 <= comparison["coverage"] <= 1


class TestSampleData:
    """Tests for sample data files."""

    def test_sample_files_exist(self):
        sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_data"
        )
        if os.path.exists(sample_dir):
            files = os.listdir(sample_dir)
            assert len(files) >= 3  # At least some sample files

    def test_sample_files_not_empty(self):
        sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sample_data"
        )
        if os.path.exists(sample_dir):
            for filename in os.listdir(sample_dir):
                path = os.path.join(sample_dir, filename)
                if os.path.isfile(path):
                    assert os.path.getsize(path) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
