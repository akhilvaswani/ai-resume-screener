"""
AI Resume Screener - Report Generator
Generates detailed screening reports in JSON and CSV formats.
"""

import os
import json
import csv
from datetime import datetime


class ReportGenerator:
    """Generates screening reports for individual and batch results."""

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_individual_report(self, result, resume_name, job_title=""):
        """
        Generate a detailed report for a single resume screening.

        Args:
            result: The screening result dict from ResumeScreener
            resume_name: Name/identifier for the resume
            job_title: Title of the job position

        Returns:
            Path to the saved report
        """
        report = {
            "metadata": {
                "resume": resume_name,
                "job_title": job_title,
                "screened_at": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "scores": {
                "overall": result["overall_score"],
                "semantic_similarity": result["semantic_score"],
                "skill_coverage": result["skill_score"],
                "category_breakdown": result.get("category_scores", {})
            },
            "skills": {
                "matched": result["matched_skills"],
                "missing": result["missing_skills"],
                "bonus": result["bonus_skills"],
                "resume_skills": result["resume_skills"],
                "required_skills": result["job_skills"]
            },
            "recommendation": result["recommendation"]
        }

        # Save JSON report
        safe_name = self._safe_filename(resume_name)
        json_path = os.path.join(self.output_dir, f"{safe_name}_report.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return json_path

    def generate_batch_report(self, results, job_title=""):
        """
        Generate a summary report for batch screening results.

        Args:
            results: List of (resume_name, screening_result) tuples
            job_title: Title of the job position

        Returns:
            Paths to the generated reports (csv_path, json_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sort by overall score descending
        sorted_results = sorted(
            results,
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )

        # CSV report
        csv_path = os.path.join(
            self.output_dir, f"batch_report_{timestamp}.csv"
        )
        self._write_csv_report(sorted_results, csv_path, job_title)

        # JSON report
        json_path = os.path.join(
            self.output_dir, f"batch_report_{timestamp}.json"
        )
        self._write_json_report(sorted_results, json_path, job_title)

        return csv_path, json_path

    def _write_csv_report(self, results, path, job_title):
        """Write a CSV summary of batch results."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Rank", "Candidate", "Overall Score", "Semantic Score",
                "Skill Score", "Matched Skills", "Missing Skills",
                "Bonus Skills", "Recommendation"
            ])

            for i, (name, result) in enumerate(results, 1):
                writer.writerow([
                    i,
                    name,
                    f"{result['overall_score']:.1%}",
                    f"{result['semantic_score']:.1%}",
                    f"{result['skill_score']:.1%}",
                    len(result["matched_skills"]),
                    len(result["missing_skills"]),
                    len(result["bonus_skills"]),
                    result["recommendation"]["level"]
                ])

    def _write_json_report(self, results, path, job_title):
        """Write a detailed JSON report of batch results."""
        report = {
            "metadata": {
                "job_title": job_title,
                "total_candidates": len(results),
                "screened_at": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "summary": {
                "strong_matches": sum(
                    1 for _, r in results
                    if r["recommendation"]["level"] == "Strong Match"
                ),
                "potential_matches": sum(
                    1 for _, r in results
                    if r["recommendation"]["level"] == "Potential Match"
                ),
                "not_recommended": sum(
                    1 for _, r in results
                    if r["recommendation"]["level"] == "Not Recommended"
                ),
                "average_score": sum(
                    r["overall_score"] for _, r in results
                ) / len(results) if results else 0
            },
            "rankings": [
                {
                    "rank": i,
                    "candidate": name,
                    "overall_score": result["overall_score"],
                    "semantic_score": result["semantic_score"],
                    "skill_score": result["skill_score"],
                    "matched_skills": result["matched_skills"],
                    "missing_skills": result["missing_skills"],
                    "recommendation": result["recommendation"]
                }
                for i, (name, result) in enumerate(results, 1)
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _safe_filename(self, name):
        """Convert a name to a safe filename."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return safe.strip("_")[:100]

    def print_summary(self, results):
        """Print a formatted summary of batch results to console."""
        sorted_results = sorted(
            results,
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )

        print(f"\n{'=' * 70}")
        print(f"SCREENING RESULTS")
        print(f"{'=' * 70}")
        print(f"{'Rank':<6}{'Candidate':<30}{'Score':<10}{'Recommendation':<20}")
        print(f"{'-' * 70}")

        for i, (name, result) in enumerate(sorted_results, 1):
            score = f"{result['overall_score']:.0%}"
            rec = result["recommendation"]["level"]
            print(f"{i:<6}{name[:28]:<30}{score:<10}{rec:<20}")

        print(f"{'=' * 70}")

        # Summary stats
        strong = sum(1 for _, r in results
                     if r["recommendation"]["level"] == "Strong Match")
        potential = sum(1 for _, r in results
                       if r["recommendation"]["level"] == "Potential Match")
        not_rec = sum(1 for _, r in results
                      if r["recommendation"]["level"] == "Not Recommended")

        print(f"\nTotal candidates: {len(results)}")
        print(f"Strong matches: {strong}")
        print(f"Potential matches: {potential}")
        print(f"Not recommended: {not_rec}")
