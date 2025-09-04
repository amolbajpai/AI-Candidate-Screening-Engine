# 📄 AI Candidate Screening Engine

An **AI-powered resume screening system** built with **FastAPI**, **LangChain**, **LangGraph**, and **Google Gemini LLM**.  
It automatically extracts information from resumes (PDF/DOCX), evaluates candidates against a Job Description (JD) using **ATS scoring methodology**, and generates an **Excel report** with ranked candidates.

---

## 🚀 Features
- 📑 **Resume Parsing** – Extracts structured information (name, email, phone, skills, education, experience) from resumes.  
- 🤖 **LLM-powered Data Extraction** – Uses **Gemini 1.5 Flash** for structured JSON extraction.  
- 🏆 **ATS Scoring** – Scores candidates based on:
  - Skills Match (40%)  
  - Relevant Experience (30%)  
  - Education Match (10%)  
  - Role-Specific Keywords (10%)  
  - Resume Structure/Clarity (10%)  
- 📊 **Excel Export** – Generates a clean, auto-formatted Excel report with top candidates sorted by ATS score.  
- 🌐 **REST API** – Endpoints for uploading resumes, downloading reports, and health checks.  
- ⚡ **Single-Step Processing** – Upload multiple resumes + JD in one API call.  

---

## 🛠️ Tech Stack
- **Backend**: FastAPI  
- **AI/LLM**: Google Gemini (via `langchain_google_genai`)  
- **Workflow Engine**: LangGraph  
- **Parsing**: LangChain `PDFPlumberLoader`, `Docx2txtLoader`  
- **Data Handling**: Pandas  
- **Output Parsing**: Pydantic + OutputFixingParser  

---

## 📂 Project Structure
```

├── main.py              # FastAPI app (Resume Shortlister API)
├── temp_uploads/        # Temporary folder for uploaded resumes & Excel outputs
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

````

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/amolbajpai/AI-Candidate-Screening-Engine.git
cd AI-Candidate-Screening-Engine
````

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variable

You need a valid **Google API Key** for Gemini.

```bash
export GOOGLE_API_KEY="your_google_api_key"   # Linux/Mac
set GOOGLE_API_KEY="your_google_api_key"      # Windows (cmd)
```

---

## ▶️ Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Server will be available at:

* API Root: [http://localhost:8000](http://localhost:8000)
* Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc Docs: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Endpoints

### 1. Health Check

**`GET /health`**

```json
{
  "status": "healthy",
  "timestamp": "2025-09-05T12:34:56"
}
```

---

### 2. Upload & Process Resumes

**`POST /upload_and_process`**

* Accepts:

  * `job_description` (string, required)
  * `required_experience` (int, required)
  * `files` (list of resumes in PDF/DOCX format)

**Example (cURL):**

```bash
curl -X POST "http://localhost:8000/upload_and_process" \
  -F "job_description=Python Developer with FastAPI experience" \
  -F "required_experience=3" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx"
```

**Response Example:**

```json
{
  "processed_count": 2,
  "excel_filename": "resume_shortlist_20250905_123456.xlsx",
  "top_candidates": [
    {
      "ats_score": 85,
      "candidate_name": "John Doe",
      "total_experience": "5",
      "relevant_experience": "3",
      "skills_match_percentage": 78,
      "contact_number": "+91-9876543210",
      "email_id": "john.doe@example.com",
      "linkedin_profile": "linkedin.com/in/johndoe",
      "summary": "Strong Python developer with FastAPI experience",
      "filename": "resume1.pdf"
    }
  ]
}
```

---

### 3. Download Excel Report

**`GET /download_excel/{filename}`**
Downloads the generated **Excel shortlist**.

Example:

```
http://localhost:8000/download_excel/resume_shortlist_20250905_123456.xlsx
```

---

## 📊 Output Example

Excel file contains:

| ATS Score | Candidate Name | Resume File | Total Experience | Relevant Experience | Skills Match % | Contact Number | Email ID                                    | LinkedIn                | Summary                                         |
| --------- | -------------- | ----------- | ---------------- | ------------------- | -------------- | -------------- | ------------------------------------------- | ----------------------- | ----------------------------------------------- |
| 90        | John Doe       | resume1.pdf | 5                | 3                   | 78%            | +91-9876543210 | [john@example.com](mailto:john@example.com) | linkedin.com/in/johndoe | Strong Python developer with FastAPI experience |

---

## 🧩 Future Improvements

* Add support for more resume formats (TXT, HTML).
* Improve skill extraction with embeddings.
* Add UI (Streamlit/React) for easier usage.
* Multi-language resume parsing.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and create a PR.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

Developed by **Amol Bajpai**
📧 [amolbajpai10@gmail.com](mailto:amolbajpai10@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/amol-bajpai) | [GitHub](https://github.com/amolbajpai)

```

Would you like me to also generate a **`requirements.txt`** file for this project so the README setup becomes fully ready to run?
```
