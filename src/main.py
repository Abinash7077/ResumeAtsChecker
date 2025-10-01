import os
import json
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path
from tempfile import NamedTemporaryFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
import asyncio

# Load .env
load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GENAI_API_KEY environment variable not set")

# Configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "https://resumeatschecker-1-eeh8.onrender.com",
        "https://ats-resume-frontend-phi.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PDF text extractor
def extract_text_from_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])

# Text splitter
def split_text_into_chunks(text, chunk_size=2500, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

# Prompt generator
def make_prompt(resumeText, job_desc="", role=""):
    if job_desc:
        jd_text = job_desc
    elif role:
        jd_text = f"Analyze this resume for the role of '{role}'. Use relevant market-standard keywords and ATS criteria."
    else:
        jd_text = "No job description or role provided. Evaluate the resume based on market-standard ATS keywords and criteria."

    return f"""
You are a resume parsing assistant. The user will provide a resume and optionally a job description or role.
If job description is provided, analyze the resume against it.
If no job description but role is provided, use market-standard keywords relevant to the role.
If neither is provided, perform a generic ATS analysis based on market standards.
Your tasks:
1. **Skills**: Identify a 'Skills' section. Categorize the skills as technical (e.g., programming languages, tools) or soft skills (e.g., communication, leadership). If a 'Skills' section is missing or not properly categorized, suggest improvements.
2. **Education**: Extract the degree, institution, and dates from the 'Education' section. If incomplete or misformatted, suggest improvements.
3. **Experience**: Evaluate the experience descriptions and ensure measurable achievements are highlighted (e.g., "Improved application performance by X%").
4. **Certifications**: Identify any certifications and verify if they are listed properly with issuing organizations and dates.
5. **Contact Information**: Ensure email and phone number are present and correctly formatted.
6. **ATS Keywords**: Compare the resume content against the job description or role keywords or ats analysis based on market standards..
Return ONLY valid JSON (no explanation outside JSON). The JSON must contain these top-level keys:
- name (String or null)
- contact (object: email, phone if available else null)
- skills (list of strings)
- experience (list of objects with {{"role","company","start_date","end_date","description","location"}})
- education (list of objects with {{"degree","institution","start_date","end_date"}}) (use "continuing" if ongoing)
- certifications (list of objects with {{"name","issuing_organization","issue_date","expiration_date","certificate_link"}})
- ats_score (number between 0-100)
- keywords (list of strings)
- suggestions (list of strings)
- ats_reason (short string explaining the score and keyword matches/mismatches, good and bad sections)

Resume Text:
\"\"\"{resumeText}\"\"\"

Job Description or Role:
\"\"\"{jd_text}\"\"\"

Return only JSON response.
"""

# Merge chunk results
def merge_results(results):
    merged = {
        "name": None,
        "contact": {"email": None, "phone": None},
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "ats_score": 0,
        "keywords": [],
        "suggestions": [],
        "ats_reason": "",
    }

    for r in results:
        for key in ["skills", "experience", "education", "certifications", "keywords", "suggestions"]:
            merged[key].extend(r.get(key, []))
        if not merged["name"] and r.get("name"):
            merged["name"] = r["name"]
        if not merged["contact"]["email"] and r.get("contact", {}).get("email"):
            merged["contact"]["email"] = r["contact"]["email"]
        if not merged["contact"]["phone"] and r.get("contact", {}).get("phone"):
            merged["contact"]["phone"] = r["contact"]["phone"]

    scores = [r.get("ats_score", 0) for r in results if isinstance(r.get("ats_score"), (int, float))]
    merged["ats_score"] = sum(scores) // len(scores) if scores else 0

    # Deduplicate lists
    for key in ["skills", "keywords", "suggestions"]:
        merged[key] = list(dict.fromkeys(merged[key]))

    return merged

# Async chunk processing
async def process_chunk(chunk, job_description=""):
    prompt = make_prompt(chunk, job_description)
    response = model.generate_content(prompt)
    raw_text = response.text
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    json_text = match.group(0) if match else raw_text.strip()
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        parsed = {"raw_output": raw_text, "error": "Failed to parse JSON"}
    return parsed

async def process_chunks_concurrent(chunks, job_description=""):
    tasks = [process_chunk(chunk, job_description) for chunk in chunks]
    return await asyncio.gather(*tasks)

# POST endpoint with async chunk processing
@app.post("/parse_resume")
async def parse_resume(file: UploadFile = File(...), job_description: str = Form("")):
    with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        resume_text = extract_text_from_pdf(tmp_path)
        chunks = split_text_into_chunks(resume_text)
        chunk_results = await process_chunks_concurrent(chunks, job_description)
        merged_result = merge_results(chunk_results)

        return JSONResponse(content={"ok": True, "data": merged_result})
    finally:
        os.remove(tmp_path)
