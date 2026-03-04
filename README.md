# AI Resume Screener

Built this to mess around with OpenAI's text embeddings and see if they could do a better job matching resumes to job descriptions than basic keyword search. Turns out they can -- keyword matching misses a ton of good candidates just because they word things differently.

The idea is pretty simple: take a resume and a job description, convert both into vector embeddings, and compare how similar they are. So if a job wants "cloud infrastructure experience" and someone's resume says "deployed apps on AWS EC2," the system gets that those are related even though the words don't match.

## What it does

- Scores resumes against job descriptions using cosine similarity on OpenAI embeddings
- Pulls out specific skills from both documents and compares them separately
- Gives you a breakdown of what matched, what's missing, and any bonus skills
- Can process a whole folder of resumes at once with `batch_screener.py`
- Has a basic Streamlit UI if you don't want to use the command line

## How scoring works

The final score is 60% semantic similarity (how well the overall content matches) and 40% skill overlap (what percentage of required skills are actually in the resume). I landed on those weights after testing a bunch of different combos against resumes where I already knew the answer.

The skill extraction piece (`skill_extractor.py`) handles stuff like "JS" vs "JavaScript" and "K8s" vs "Kubernetes" using fuzzy matching, which was more annoying to get right than I expected.

## Running it

You'll need an OpenAI API key in your `.env` file. There's a `.env.example` to get started.

```bash
pip install -r requirements.txt

# screen a single resume
python screener.py --resume resume.txt --job job_description.txt

# batch mode
python batch_screener.py --job job_description.txt --resumes ./resume_folder/

# web UI
streamlit run app.py
```

Embeddings get cached in a local SQLite database so you're not paying for the same text twice.

## Files

- `screener.py` -- main screening logic
- `embeddings.py` -- handles OpenAI API calls + caching
- `skill_extractor.py` -- pulls skills out of text and categorizes them
- `batch_screener.py` -- processes multiple resumes at once
- `report_generator.py` -- exports results as JSON/CSV
- `app.py` -- Streamlit web interface
- `sample_data/` -- some example resumes and job descriptions to test with
