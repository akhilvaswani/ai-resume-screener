"""
AI Resume Screener - Batch Processing
Screen multiple resumes against a job description and rank candidates.
"""

import os
import argparse

from screener import ResumeScreener
from report_generator import ReportGenerator


SUPPORTED_EXTENSIONS = {".txt", ".md", ".text"}


def load_text_file(file_path):
    """Read a text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def find_resumes(resume_dir):
    """Find all supported resume files in a directory."""
    resumes = []
    for filename in sorted(os.listdir(resume_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            resumes.append(os.path.join(resume_dir, filename))
    return resumes


def main():
    parser = argparse.ArgumentParser(
        description="Screen multiple resumes against a job description"
    )
    parser.add_argument(
        "--job", type=str, required=True,
        help="Path to the job description file"
    )
    parser.add_argument(
        "--resumes", type=str, required=True,
        help="Path to directory containing resume files"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for reports (default: results)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum score threshold (0.0-1.0) to include in results"
    )
    parser.add_argument(
        "--top-n", type=int, default=0,
        help="Only show top N candidates (0 = show all)"
    )
    parser.add_argument(
        "--job-title", type=str, default="",
        help="Job title for the report header"
    )
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.job):
        print(f"Error: Job description not found: {args.job}")
        return

    if not os.path.isdir(args.resumes):
        print(f"Error: Resume directory not found: {args.resumes}")
        return

    # Load job description
    print(f"Loading job description: {args.job}")
    job_text = load_text_file(args.job)

    # Find resumes
    resume_files = find_resumes(args.resumes)
    if not resume_files:
        print(f"No supported resume files found in: {args.resumes}")
        return

    print(f"Found {len(resume_files)} resumes to screen")

    # Initialize screener and report generator
    screener = ResumeScreener()
    reporter = ReportGenerator(output_dir=args.output)

    # Screen each resume
    results = []
    for i, resume_path in enumerate(resume_files):
        filename = os.path.basename(resume_path)
        name = os.path.splitext(filename)[0]
        print(f"\n[{i + 1}/{len(resume_files)}] Screening: {filename}")

        resume_text = load_text_file(resume_path)
        result = screener.screen(resume_text, job_text)

        # Apply threshold filter
        if result["overall_score"] >= args.threshold:
            results.append((name, result))

            # Generate individual report
            reporter.generate_individual_report(
                result, name, job_title=args.job_title
            )

            print(f"  Score: {result['overall_score']:.0%} - "
                  f"{result['recommendation']['level']}")
            print(f"  Matched: {len(result['matched_skills'])} skills | "
                  f"Missing: {len(result['missing_skills'])} skills")
        else:
            print(f"  Score: {result['overall_score']:.0%} - "
                  f"Below threshold ({args.threshold:.0%})")

    if not results:
        print("\nNo candidates met the threshold criteria.")
        return

    # Apply top-n filter
    if args.top_n > 0:
        results = sorted(
            results, key=lambda x: x[1]["overall_score"], reverse=True
        )[:args.top_n]

    # Generate batch report
    csv_path, json_path = reporter.generate_batch_report(
        results, job_title=args.job_title
    )

    # Print summary
    reporter.print_summary(results)

    print(f"\nReports saved to: {args.output}/")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
