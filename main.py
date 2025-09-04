import os
import json
import shutil
from datetime import datetime
from typing import List, Dict
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Document processing
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
import pandas as pd
import re

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=2048
)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Shortlister API",
    description="AI-powered resume screening system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for uploaded files
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models
class ProcessingResult(BaseModel):
    processed_count: int
    excel_filename: str
    top_candidates: List[Dict]

# LangGraph State Definition
class ResumeProcessingState(TypedDict):
    job_description: str
    required_experience: int
    resume_files: List[str]
    raw_texts: Dict[str, str]
    extracted_data: Dict[str, Dict]
    scored_data: Dict[str, Dict]
    final_results: List[Dict]
    excel_filename: str


class CandidateDetails(BaseModel):
    candidate_name: str
    total_experience: float
    education: str
    contact_number: str
    email_id: str
    linkedin_profile: str
    skills: List[str]
    experience_details: str




class ResumeProcessor:
    """Main class for processing resumes using LangChain and LangGraph"""
    
    def __init__(self):
        self.llm = llm
        self.setup_prompts()
        self.setup_langgraph()
    
    def setup_prompts(self):
        """Set up LangChain prompt templates"""
        # ✅ Base parser
        parser = PydanticOutputParser(pydantic_object=CandidateDetails)

        # ✅ Wrap with OutputFixingParser (normal usage)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        self.extraction_prompt = PromptTemplate(
            input_variables=["resume_text"],
            template="""
            Analyze the following resume and extract structured information. If any field is not found, return "Not Found".
            
            Resume Text:
            {resume_text}
            
            {format_instructions}
            }}
            
            Return only valid JSON, no additional text.
            """,
            partial_variables={"format_instructions": fixing_parser.get_format_instructions()}
        )
        
        self.scoring_prompt = PromptTemplate(
            input_variables=["candidate_data", "job_description", "required_experience"],
            template="""
            Score this candidate against the job requirements using ATS scoring methodology.
            
            Candidate Data:
            {candidate_data}
            
            Job Description:
            {job_description}
            
            Required Experience: {required_experience} years
            
            Score based on:
            1. Skills Match (40%) - How many JD skills match candidate skills
            2. Relevant Experience (30%) - Years of relevant experience vs required
            3. Education Match (10%) - Education alignment with job requirements  
            4. Role-Specific Keywords (10%) - Important keywords from JD found in resume
            5. Resume Structure/Clarity (10%) - Completeness of information
            
            Calculate relevant experience years based on the candidate's work history alignment with the job role.
            
            Return ONLY a JSON object:
            {{
                "ats_score": 85,
                "relevant_experience": "3",
                "skills_match_percentage": 75,
                "summary": "Brief 2-3 sentence summary of candidate suitability"
            }}
            
            Return only valid JSON, no additional text.
            """
        )
    
    def setup_langgraph(self):
        """Set up LangGraph workflow"""
        workflow = StateGraph(ResumeProcessingState)
        
        # Add nodes
        workflow.add_node("parse_resumes", self.parse_resumes_node)
        workflow.add_node("extract_data", self.extract_data_node)
        workflow.add_node("score_candidates", self.score_candidates_node)
        workflow.add_node("generate_summary", self.generate_summary_node)
        workflow.add_node("export_excel", self.export_excel_node)
        
        # Define the flow
        workflow.set_entry_point("parse_resumes")
        workflow.add_edge("parse_resumes", "extract_data")
        workflow.add_edge("extract_data", "score_candidates")
        workflow.add_edge("score_candidates", "generate_summary")
        workflow.add_edge("generate_summary", "export_excel")
        workflow.set_finish_point("export_excel")
        
        self.graph = workflow.compile()
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file using LangChain PDFPlumberLoader"""
        try:
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents]).strip()
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file using LangChain Docx2txtLoader"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents]).strip()
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
            return ""
    
    def parse_resumes_node(self, state: ResumeProcessingState) -> ResumeProcessingState:
        """LangGraph node: Parse resume files and extract text"""
        raw_texts = {}
        
        for file_path in state["resume_files"]:
            filename = os.path.basename(file_path)
            
            try:
                if file_path.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                elif file_path.lower().endswith('.docx'):
                    text = self.extract_text_from_docx(file_path)
                else:
                    text = "Unsupported file format"
                
                raw_texts[filename] = text
                print(f"Successfully extracted text from {filename}: {len(text)} characters")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                raw_texts[filename] = ""
        
        state["raw_texts"] = raw_texts
        return state
    
    def extract_data_node(self, state: ResumeProcessingState) -> ResumeProcessingState:
        """LangGraph node: Extract structured data using LLM"""
        extracted_data = {}
        
        for filename, text in state["raw_texts"].items():
            if not text or text == "Unsupported file format":
                extracted_data[filename] = self.create_default_structure()
                continue
            
            try:
                prompt = self.extraction_prompt.format(resume_text=text)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                json_str = self.clean_json_response(response.content)
                extracted_info = json.loads(json_str)
                extracted_data[filename] = extracted_info
                
            except Exception as e:
                print(f"Error extracting data from {filename}: {e}")
                extracted_data[filename] = self.manual_fallback_extraction(text)
        
        state["extracted_data"] = extracted_data
        return state
    
    def create_default_structure(self) -> Dict:
        """Create default structure for failed extractions"""
        return {
            "candidate_name": "Not Found",
            "total_experience": "Not Found",
            "education": "Not Found",
            "contact_number": "Not Found",
            "email_id": "Not Found",
            "linkedin_profile": "Not Found",
            "skills": [],
            "experience_details": "Not Found"
        }
    
    def clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response"""
        json_str = response.strip()
        if json_str.startswith('```json'):
            json_str = json_str[7:-3]
        elif json_str.startswith('```'):
            json_str = json_str[3:-3]
        return json_str
    
    def manual_fallback_extraction(self, text: str) -> Dict:
        """Fallback manual extraction if LLM fails"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        linkedin_pattern = r'linkedin\.com/in/[A-Za-z0-9-]+'
        
        email_match = re.search(email_pattern, text)
        phone_match = re.search(phone_pattern, text)
        linkedin_match = re.search(linkedin_pattern, text)
        
        return {
            "candidate_name": "Not Found",
            "total_experience": "Not Found", 
            "education": "Not Found",
            "contact_number": phone_match.group() if phone_match else "Not Found",
            "email_id": email_match.group() if email_match else "Not Found",
            "linkedin_profile": linkedin_match.group() if linkedin_match else "Not Found",
            "skills": [],
            "experience_details": "Not Found"
        }
    
    def score_candidates_node(self, state: ResumeProcessingState) -> ResumeProcessingState:
        """LangGraph node: Score candidates using ATS methodology"""
        scored_data = {}
        
        for filename, candidate_data in state["extracted_data"].items():
            try:
                candidate_json = json.dumps(candidate_data, indent=2)
                prompt = self.scoring_prompt.format(
                    candidate_data=candidate_json,
                    job_description=state["job_description"],
                    required_experience=state["required_experience"]
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                json_str = self.clean_json_response(response.content)
                scoring_result = json.loads(json_str)
                
                final_candidate_data = candidate_data.copy()
                final_candidate_data.update(scoring_result)
                scored_data[filename] = final_candidate_data
                
            except Exception as e:
                print(f"Error scoring candidate {filename}: {e}")
                final_candidate_data = candidate_data.copy()
                final_candidate_data.update({
                    "ats_score": 0,
                    "relevant_experience": "Not Found",
                    "skills_match_percentage": 0,
                    "summary": "Unable to process candidate data"
                })
                scored_data[filename] = final_candidate_data
        
        state["scored_data"] = scored_data
        return state
    
    def generate_summary_node(self, state: ResumeProcessingState) -> ResumeProcessingState:
        """LangGraph node: Generate final summary and structure data"""
        final_results = []
        
        for filename, candidate_data in state["scored_data"].items():
            result = {
                "ats_score": candidate_data.get("ats_score", 0),
                "candidate_name": candidate_data.get("candidate_name", "Not Found"),
                "total_experience": candidate_data.get("total_experience", "Not Found"),
                "relevant_experience": candidate_data.get("relevant_experience", "Not Found"),
                "skills_match_percentage": candidate_data.get("skills_match_percentage", 0),
                "contact_number": candidate_data.get("contact_number", "Not Found"),
                "email_id": candidate_data.get("email_id", "Not Found"),
                "linkedin_profile": candidate_data.get("linkedin_profile", "Not Found"),
                "summary": candidate_data.get("summary", "No summary available"),
                "filename": filename
            }
            final_results.append(result)
        
        # Sort by ATS score (highest first)
        final_results.sort(key=lambda x: x["ats_score"], reverse=True)
        state["final_results"] = final_results
        return state
    
    def export_excel_node(self, state: ResumeProcessingState) -> ResumeProcessingState:
        """LangGraph node: Export results to Excel"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resume_shortlist_{timestamp}.xlsx"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        df_data = []
        for result in state["final_results"]:
            df_data.append({
                "ATS Score": result["ats_score"],
                "Candidate Name": result["candidate_name"],
                "Resume File": result["filename"],  # Added Resume File column
                "Total Experience": result["total_experience"],
                "Relevant Experience": result["relevant_experience"],
                "Skills Match %": result["skills_match_percentage"],
                "Contact Number": result["contact_number"],
                "Email ID": result["email_id"],
                "LinkedIn": result["linkedin_profile"],
                "Summary": result["summary"]
            })
        
        df = pd.DataFrame(df_data)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Resume Shortlist')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Resume Shortlist']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        state["excel_filename"] = filename
        return state
    
    async def process_resumes(self, job_description: str, required_experience: int, file_paths: List[str]) -> Dict:
        """Main processing function using LangGraph"""
        initial_state = ResumeProcessingState(
            job_description=job_description,
            required_experience=required_experience,
            resume_files=file_paths,
            raw_texts={},
            extracted_data={},
            scored_data={},
            final_results=[],
            excel_filename=""
        )
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "processed_count": len(final_state["final_results"]),
            "excel_filename": final_state["excel_filename"],
            "results": final_state["final_results"]
        }

# Initialize the processor
processor = ResumeProcessor()

# Single Combined Endpoint
@app.post("/upload_and_process", response_model=ProcessingResult)
async def upload_and_process_resumes(
    job_description: str = Form(...),
    required_experience: int = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload resumes and process them in a single endpoint
    """
    try:
        # Clear previous uploads
        for file in os.listdir(UPLOAD_DIR):
            if file.endswith(('.pdf', '.docx')):
                os.remove(os.path.join(UPLOAD_DIR, file))
        
        uploaded_files = []
        
        # Save uploaded files
        for file in files:
            if not file.filename.lower().endswith(('.pdf', '.docx')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format: {file.filename}. Only PDF and DOCX are supported."
                )
            
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_files.append(file_path)
        
        # Process resumes immediately
        result = await processor.process_resumes(
            job_description=job_description,
            required_experience=required_experience,
            file_paths=uploaded_files
        )
        
        return ProcessingResult(
            processed_count=result["processed_count"],
            excel_filename=result["excel_filename"],
            top_candidates=result["results"][:5]  # Return top 5 for preview
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/download_excel/{filename}")
async def download_excel(filename: str):
    """Download the generated Excel report"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Resume Shortlister API",
        "version": "1.0.0",
        "endpoints": {
            "upload_and_process": "/upload_and_process",
            "download": "/download_excel/{filename}",
            "health": "/health"
        },
        "description": "AI-powered resume screening system using LangChain, LangGraph, and Gemini LLM"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Resume Shortlister API...")
    print("Make sure to set your GOOGLE_API_KEY environment variable")
    print("API will be available at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )