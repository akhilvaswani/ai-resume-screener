"""
AI Resume Screener - Skill Extraction
Extracts and categorizes skills from resumes and job descriptions
using NLP techniques and a curated skill taxonomy.
"""

import re


# Skill taxonomy organized by category
SKILL_TAXONOMY = {
    "programming": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go",
        "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R",
        "MATLAB", "Perl", "Shell", "Bash", "PowerShell", "SQL", "HTML",
        "CSS", "Dart", "Lua", "Haskell", "Elixir"
    ],
    "cloud": [
        "AWS", "Amazon Web Services", "Azure", "Microsoft Azure",
        "Google Cloud", "GCP", "Google Cloud Platform",
        "EC2", "S3", "Lambda", "ECS", "EKS", "RDS", "DynamoDB",
        "CloudFormation", "IAM", "VPC", "Route 53", "CloudWatch",
        "Azure Functions", "Azure DevOps", "Azure AD",
        "Cloud Run", "BigQuery", "Cloud Storage",
        "Heroku", "DigitalOcean", "Linode"
    ],
    "databases": [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "SQLite", "Oracle", "SQL Server", "Cassandra", "DynamoDB",
        "Neo4j", "CouchDB", "MariaDB", "Memcached", "InfluxDB",
        "Snowflake", "Redshift", "BigQuery"
    ],
    "frameworks": [
        "React", "Angular", "Vue", "Node.js", "Express", "Django",
        "Flask", "FastAPI", "Spring", "Spring Boot", ".NET",
        "Rails", "Ruby on Rails", "Laravel", "Next.js", "Nuxt",
        "Svelte", "TensorFlow", "PyTorch", "Keras", "scikit-learn",
        "Pandas", "NumPy", "Streamlit", "Gradio"
    ],
    "tools": [
        "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins",
        "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI",
        "Prometheus", "Grafana", "Datadog", "Splunk", "ELK Stack",
        "Nginx", "Apache", "Git", "SVN", "Jira", "Confluence",
        "Slack", "Postman", "Swagger", "Kafka", "RabbitMQ",
        "Airflow", "dbt", "Spark", "Hadoop", "Tableau", "Power BI",
        "Wireshark", "Nmap"
    ],
    "practices": [
        "CI/CD", "DevOps", "Agile", "Scrum", "Kanban",
        "TDD", "test-driven development", "BDD",
        "microservices", "REST", "RESTful", "GraphQL", "gRPC",
        "serverless", "Infrastructure as Code", "IaC",
        "containerization", "orchestration", "monitoring",
        "logging", "observability", "incident response",
        "disaster recovery", "high availability", "load balancing",
        "caching", "message queues", "event-driven",
        "machine learning", "deep learning", "NLP",
        "computer vision", "data engineering", "data pipeline",
        "ETL", "data warehousing", "data modeling"
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "collaboration",
        "problem solving", "critical thinking", "time management",
        "project management", "mentoring", "coaching",
        "presentation", "stakeholder management",
        "cross-functional", "agile methodology"
    ],
    "certifications": [
        "AWS Certified", "Azure Certified", "GCP Certified",
        "Certified Kubernetes", "CKA", "CKAD",
        "CompTIA", "Security+", "Network+", "A+",
        "CISSP", "CEH", "CCNA", "CCNP",
        "PMP", "Scrum Master", "CSM",
        "Terraform Associate", "HashiCorp Certified"
    ]
}


class SkillExtractor:
    """
    Extracts skills from text using taxonomy matching,
    n-gram analysis, and fuzzy matching.
    """

    def __init__(self, custom_taxonomy=None):
        self.taxonomy = custom_taxonomy or SKILL_TAXONOMY
        self._build_lookup()

    def _build_lookup(self):
        """Build a fast lookup structure for skill matching."""
        self.skill_to_category = {}
        self.lowercase_lookup = {}

        for category, skills in self.taxonomy.items():
            for skill in skills:
                self.skill_to_category[skill] = category
                self.lowercase_lookup[skill.lower()] = (skill, category)

    def extract(self, text):
        """
        Extract skills from text and organize by category.

        Args:
            text: The resume or job description text

        Returns:
            dict mapping category names to lists of found skills
        """
        if not text:
            return {}

        found_skills = {}
        text_lower = text.lower()

        # Check each skill in the taxonomy
        for skill_lower, (skill_original, category) in self.lowercase_lookup.items():
            # Use word boundary matching for short skills to avoid false positives
            if len(skill_lower) <= 3:
                pattern = r"\b" + re.escape(skill_lower) + r"\b"
                if re.search(pattern, text_lower):
                    if category not in found_skills:
                        found_skills[category] = []
                    if skill_original not in found_skills[category]:
                        found_skills[category].append(skill_original)
            else:
                if skill_lower in text_lower:
                    if category not in found_skills:
                        found_skills[category] = []
                    if skill_original not in found_skills[category]:
                        found_skills[category].append(skill_original)

        # Also extract years of experience mentions
        experience = self._extract_experience(text)
        if experience:
            found_skills["experience"] = experience

        return found_skills

    def _extract_experience(self, text):
        """Extract years of experience mentions."""
        patterns = [
            r"(\d+)\+?\s*years?\s*(?:of\s+)?experience",
            r"(\d+)\+?\s*years?\s*(?:of\s+)?(?:professional\s+)?experience",
            r"experience\s*(?:of\s+)?(\d+)\+?\s*years?",
        ]

        experiences = []
        text_lower = text.lower()

        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                years = int(match.group(1))
                context = text_lower[max(0, match.start() - 50):match.end() + 50]
                experiences.append(f"{years}+ years")

        return list(set(experiences)) if experiences else []

    def get_all_skills_flat(self, text):
        """Extract all skills as a flat list (no categories)."""
        categorized = self.extract(text)
        all_skills = []
        for skills in categorized.values():
            all_skills.extend(skills)
        return sorted(set(all_skills))

    def compare_skills(self, resume_text, job_text):
        """
        Compare skills between a resume and job description.

        Returns:
            dict with matched, missing, and bonus skills
        """
        resume_skills = set(s.lower() for s in self.get_all_skills_flat(resume_text))
        job_skills = set(s.lower() for s in self.get_all_skills_flat(job_text))

        return {
            "matched": sorted(resume_skills & job_skills),
            "missing": sorted(job_skills - resume_skills),
            "bonus": sorted(resume_skills - job_skills),
            "coverage": len(resume_skills & job_skills) / len(job_skills) if job_skills else 0
        }
