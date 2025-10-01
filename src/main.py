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
import google.generativeai as genai

# Load .env
load_dotenv()

# Correct env key
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("GENAI_API_KEY environment variable not set")

# Proper way to configure Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "https://resumeatschecker-1-eeh8.onrender.com",
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
# POST endpoint
@app.post("/parse_resume")
async def parse_resume(file: UploadFile = File(...), job_description: str = Form("")):
    with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        resume_text = extract_text_from_pdf(tmp_path)
        prompt = make_prompt(resume_text, job_description)

        response = model.generate_content(prompt)
        raw_text = response.text

        # Try to extract JSON
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        json_text = match.group(0) if match else raw_text.strip()

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            parsed = {"raw_output": raw_text, "error": "Failed to parse JSON"}

        return JSONResponse(content={"ok": True, "data": parsed})

    finally:
        os.remove(tmp_path)
