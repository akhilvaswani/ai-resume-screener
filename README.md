# AI Resume Screener

Resume screening tool that matches resumes against job descriptions using OpenAI's text embeddings. Instead of relying on basic keyword matching (which misses a lot of qualified candidates), it converts both the resume and job description into vector embeddings and calculates how similar they are semantically. So if a job asks for "cloud infrastructure experience" and a resume mentions "deployed applications on AWS EC2 and managed VPC networking," the system understands those are related even though the exact words don't match.

I also built in skill extraction, a scoring breakdown that explains why each resume scored the way it did, and a batch processing mode for screening multiple resumes against the same job posting.

## Files

- `screener.py` - Core screening engine with embedding-based matching and skill extraction
- `embeddings.py` - OpenAI embedding generation with caching and rate limiting
- `skill_extractor.py` - Extracts and categorizes skills from text using NLP
- `report_generator.py` - Generates detailed screening reports in JSON and CSV
- `batch_screener.py` - Screen multiple resumes against a job description at once
- `app.py` - Streamlit web interface for interactive screening
- `config.py` - Configuration and API settings
- `sample_data/` - Example resumes and job descriptions for testing
  - `resume_software_engineer.txt` - Sample software engineer resume
  - `resume_data_analyst.txt` - Sample data analyst resume
  - `resume_cloud_engineer.txt` - Sample cloud engineer resume
  - `job_description_sre.txt` - Sample SRE job posting
  - `job_description_data_engineer.txt` - Sample data engineer job posting
- `tests/test_screener.py` - Unit tests
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variable template
- `.gitignore` - Standard Python ignores

## How I Built It

### Step 1 - Understanding the Problem

Traditional resume screening tools use keyword matching. They look for exact terms from the job description in the resume and score based on how many matches they find. The problem is this approach is really brittle. A resume that uses different terminology for the same skills gets penalized, and candidates who would be a great fit get filtered out.

Embedding-based matching solves this by converting text into high-dimensional vectors that capture semantic meaning. Two pieces of text that mean similar things will have vectors that point in similar directions, even if they use completely different words.

I went with OpenAI's `text-embedding-3-small` model because it gives a good balance of quality and cost. It produces 1536-dimensional vectors and handles up to 8191 tokens per input.

### Step 2 - Setting Up Embeddings with Caching

The `embeddings.py` module handles all interaction with the OpenAI API. Since embedding generation costs money and takes time, I built in a local cache using a SQLite database. The first time you embed a piece of text, the result gets stored. If you process the same text again (which happens a lot when you're screening multiple resumes against the same job description), it pulls from the cache instead of making another API call.

```python
from embeddings import EmbeddingManager

manager = EmbeddingManager()

# First call hits the API
vector = manager.get_embedding("Experienced Python developer with AWS skills")

# Second call with same text hits the cache
vector = manager.get_embedding("Experienced Python developer with AWS skills")
```

I also added rate limiting to stay within OpenAI's API limits, and retry logic with exponential backoff for handling transient errors.

### Step 3 - Skill Extraction

Before comparing the resume and job description as whole documents, I built a skill extraction system in `skill_extractor.py` that pulls out specific skills and categorizes them. This serves two purposes: it gives a more granular matching score, and it makes the results explainable (you can see exactly which skills matched and which were missing).

The extractor uses a combination of approaches:
1. A curated skill taxonomy organized by category (programming languages, cloud platforms, databases, frameworks, soft skills, etc.)
2. N-gram matching to catch multi-word skills like "machine learning" or "Amazon Web Services"
3. Fuzzy matching to handle variations like "JS" vs "JavaScript" or "K8s" vs "Kubernetes"
4. Context-aware extraction that distinguishes between someone who "used Python" vs someone who "managed a team of Python developers"

```python
from skill_extractor import SkillExtractor

extractor = SkillExtractor()
skills = extractor.extract(resume_text)

# Returns something like:
# {
#     "programming": ["Python", "JavaScript", "SQL"],
#     "cloud": ["AWS", "EC2", "S3", "Lambda"],
#     "databases": ["PostgreSQL", "MongoDB", "Redis"],
#     "frameworks": ["Django", "React", "Flask"],
#     "tools": ["Docker", "Kubernetes", "Terraform"],
#     "soft_skills": ["leadership", "communication"]
# }
```

### Step 4 - The Screening Pipeline

The main `ResumeScreener` class in `screener.py` combines embedding similarity with skill matching to produce a final score. Here's how it works:

1. **Parse the inputs** - Clean and preprocess both the resume and job description
2. **Extract skills** - Pull skills from both documents and categorize them
3. **Generate embeddings** - Create vector representations of:
   - The full resume text
   - The full job description text
   - Each skill category separately (so "cloud skills" in the resume gets compared to "cloud skills" in the job description)
4. **Calculate similarity scores** - Use cosine similarity to measure how close the vectors are
5. **Compute skill overlap** - Calculate what percentage of required skills the resume covers
6. **Generate a weighted final score** - Combine semantic similarity (60%) and skill overlap (40%)
7. **Produce an explanation** - Break down exactly what matched, what's missing, and what bonus skills the candidate has

```python
from screener import ResumeScreener

screener = ResumeScreener()

result = screener.screen(resume_text, job_description_text)

print(f"Overall Score: {result['overall_score']:.0%}")
print(f"Semantic Match: {result['semantic_score']:.0%}")
print(f"Skill Match: {result['skill_score']:.0%}")
print(f"Matched Skills: {result['matched_skills']}")
print(f"Missing Skills: {result['missing_skills']}")
print(f"Bonus Skills: {result['bonus_skills']}")
```

### Step 5 - Scoring Breakdown and Explainability

One thing I really wanted was for the tool to explain its scores, not just spit out a number. The screening result includes:

- **Overall score** (0-100%) - Weighted combination of semantic and skill scores
- **Semantic similarity** - How well the overall resume content matches the job description
- **Skill coverage** - What percentage of required skills are present
- **Category breakdown** - Scores for each skill category (technical, cloud, tools, etc.)
- **Matched skills** - Skills found in both the resume and job description
- **Missing skills** - Skills in the job description that aren't in the resume
- **Bonus skills** - Extra skills in the resume that weren't required but add value
- **Experience level match** - Whether the candidate's experience level fits the role
- **Recommendation** - Strong match, potential match, or not recommended, with reasoning

### Step 6 - Batch Processing

For screening multiple candidates against the same position, `batch_screener.py` processes a folder of resumes and ranks them:

```bash
python batch_screener.py \
    --job job_description.txt \
    --resumes ./resume_folder/ \
    --output ./results/
```

This produces:
- A ranked list of all candidates with their scores
- Individual detailed reports for each candidate
- A summary CSV file for easy comparison
- A visual ranking chart

I added a `--threshold` flag so you can filter out candidates below a certain score, and a `--top-n` flag to only keep the top N candidates.

### Step 7 - Streamlit Web Interface

To make this more user-friendly, I built a web interface using Streamlit in `app.py`:

```bash
streamlit run app.py
```

The interface lets you:
- Paste or upload a job description
- Upload one or more resumes (supports .txt, .pdf, and .docx)
- See results instantly with visual score breakdowns
- Compare multiple candidates side by side
- Download results as CSV or JSON

### Step 8 - Testing

The test suite in `tests/test_screener.py` covers:

- Skill extraction accuracy against known skill lists
- Embedding caching behavior
- Score consistency (same inputs should always produce the same score)
- Edge cases like empty resumes, very short job descriptions, and non-English text
- Batch processing with mixed file types

```bash
python -m pytest tests/ -v
```

## Example Results

Screening 3 sample resumes against an SRE job description:

| Candidate | Overall | Semantic | Skills | Recommendation |
|-----------|---------|----------|--------|----------------|
| Cloud Engineer Resume | 87% | 91% | 82% | Strong Match |
| Software Engineer Resume | 64% | 72% | 53% | Potential Match |
| Data Analyst Resume | 31% | 38% | 22% | Not Recommended |

The cloud engineer scored highest because their experience with AWS, Kubernetes, and monitoring tools directly aligned with the SRE requirements. The software engineer had relevant coding skills but lacked the infrastructure and operations experience. The data analyst's skill set was too different from what the role needed.

## Configuration

Settings are managed in `config.py` and `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | (from .env) | Your OpenAI API key |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `CACHE_DB` | embeddings_cache.db | SQLite cache file path |
| `SEMANTIC_WEIGHT` | 0.6 | Weight for semantic similarity in final score |
| `SKILL_WEIGHT` | 0.4 | Weight for skill overlap in final score |
| `SIMILARITY_THRESHOLD` | 0.7 | Minimum score for "Strong Match" |
| `POTENTIAL_THRESHOLD` | 0.5 | Minimum score for "Potential Match" |

## What I Learned

The most interesting part of this project was seeing how much better semantic matching is compared to keyword matching. In my testing, keyword matching missed about 35% of qualified candidates because they used different terminology. The embedding approach caught almost all of them. I also learned that combining embeddings with structured skill extraction gives much better results than either approach alone. The embeddings capture overall fit while the skill extraction provides the granular detail that makes the results actionable and explainable.
