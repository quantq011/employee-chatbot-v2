"""
FastAPI Backend for Employee Onboarding System
Integrates OpenAI (via LangChain) and ChromaDB for intelligent onboarding assistance
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from pathlib import Path
import json
import uvicorn
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
 
# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")
 
# Import the merger and embedder
from merge_template import TemplateMerger
from embedder import OnboardingEmbedder
import re
 
# Initialize FastAPI
app = FastAPI(
    title="Employee Onboarding API",
    description="AI-powered employee onboarding system with LangChain & OpenAI",
    version="2.0.0"
)
 
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Initialize OpenAI via LangChain
llm = None
try:
    # Use separate API key if provided, otherwise fall back to main key
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://aiportalapi.stu-platform.live/jpe")
   
    # Ensure endpoint ends with /v1 for OpenAI compatibility
    if not api_base.endswith('/v1'):
        api_base = api_base.rstrip('/') + '/v1'
   
    if api_key:
        # Create HTTP client with SSL verification disabled for custom endpoints
        # http_client = httpx.Client(
        #     timeout=httpx.Timeout(60.0, connect=10.0),
        #     limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        #     verify=False  # Disabled SSL verification for custom endpoints
        # )
       
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7,
            openai_api_key=api_key,
            base_url=api_base,
            #http_client=http_client
        )
        print(f"✓ OpenAI (LangChain) client initialized")
        print(f"  Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
        print(f"  Endpoint: {api_base}")
        print(f"  SSL Verification: Disabled")
        print(f"  Timeout: 60s")
    else:
        print("⚠ OPENAI_API_KEY not found. AI features will be limited.")
except Exception as e:
    print(f"⚠ Failed to initialize OpenAI: {e}")
 
# Initialize Embedder
embedder = OnboardingEmbedder(chroma_persist_dir="./chroma_db")
 
# Initialize Template Merger (path relative to project root)
merger = TemplateMerger(base_path="../documents/onboarding")
 
 
# ===========================
# Helper Functions
# ===========================
 
 
# Helper function with retry for OpenAI API calls via LangChain
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_openai_with_retry(messages):
    """Call OpenAI API (via LangChain) with automatic retry on transient failures"""
    if not llm:
        raise Exception("OpenAI client not initialized")
    return llm.invoke(messages)
 
 
# Pydantic Models
class MergeRequest(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    template_name: Optional[str] = Field(None, description="Optional: Custom template name")
    output_file: Optional[str] = Field(None, description="Optional custom output path")
    merge_sections: Optional[List[str]] = Field(None, description="Sections to merge: ['all', 'info', 'region', 'role', 'phases', 'project_specific']")
 
 
class MergeResponse(BaseModel):
    success: bool
    message: str
    output_path: Optional[str] = None
    merged_data: Optional[Dict[str, Any]] = None
 
 
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question about onboarding")
    project: Optional[str] = Field(None, description="Project context")
    role: Optional[str] = Field(None, description="Role context")
 
 
class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
 
 
class DocumentRequest(BaseModel):
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
 
 
class OnboardingStatus(BaseModel):
    employee_id: str
    project: str
    role: str
    phase: str
    completed_tasks: List[str]
    pending_tasks: List[str]
    progress_percentage: float
 
 
# API Endpoints
 
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Employee Onboarding API",
        "version": "1.0.0"
    }
 
 
@app.post("/merge", response_model=MergeResponse)
async def merge_template(request: MergeRequest):
    """
    Merge templates with project overrides
    Creates merged_config.json with role, region, and phase templates
   
    Role and region are read from project's overrides.json file
    """
    try:
        print(f"\n{'='*60}")
        print(f"MERGE REQUEST")
        print(f"{'='*60}")
        print(f"Project: {request.project_name}")
        print(f"Template Name: {request.template_name or 'from overrides.json'}")
        print(f"Merge Sections: {request.merge_sections or ['all']}")
        print(f"{'='*60}\n")
       
        # Use the new selective merge function if merge_sections is specified
        if request.merge_sections:
            merged_data = merger.merge_project_template(
                project_name=request.project_name,
                template_name=request.template_name,
                output_file=request.output_file,
                merge_sections=request.merge_sections
            )
        else:
            # Use legacy function for backward compatibility (merges all)
            merged_data = merger.merge_with_overrides(
                template_name=request.template_name or "default",
                project_name=request.project_name,
                output_file=request.output_file
            )
       
        if not merged_data:
            print(f"✗ Merge failed - no data returned")
            raise HTTPException(
                status_code=404,
                detail=f"Merge failed - check that project '{request.project_name}' exists with overrides.json"
            )
       
        output_path = request.output_file or f"documents/onboarding/projects/{request.project_name}/merged_config.json"
       
        sections_merged = merged_data.get('metadata', {}).get('merged_sections', ['all'])
        print(f"✓ Merge successful: {output_path}")
        print(f"✓ Sections merged: {', '.join(sections_merged)}\n")
       
        return MergeResponse(
            success=True,
            message=f"Successfully merged {', '.join(sections_merged)} for {request.project_name}",
            output_path=output_path,
            merged_data=merged_data
        )
   
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Merge error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Merge error: {str(e)}")
 
 
@app.get("/templates")
async def list_templates():
    """
    List all available templates
    """
    templates = merger.list_templates()
    projects = merger.list_projects()
   
    return {
        "templates": templates,
        "projects": projects
    }
 
 
@app.get("/projects/{project_name}")
async def get_project_config(project_name: str):
    """
    Get merged configuration for a specific project
    """
    config_path = Path(f"documents/onboarding/projects/{project_name}/merged_config.json")
   
    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Merged config not found for project {project_name}. Run merge first."
        )
   
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
   
    return config
 
 
@app.post("/documents/add")
async def add_document(doc: DocumentRequest):
    """
    Add a custom document to ChromaDB for RAG (deprecated - use index-project instead)
    """
    return {
        "success": False,
        "message": "This endpoint is deprecated. Use /documents/index-project instead."
    }
 
 
@app.post("/documents/index-project")
async def index_project_documents(project_name: str):
    """
    Index project configuration documents into ChromaDB with intelligent chunking
   
    Process:
    1. Load merged_config.json for the project
    2. Chunk into 7 semantic pieces (role, region, 4 phases, project-specific)
    3. Generate embeddings using Azure OpenAI text-embedding-3-small
    4. Store in ChromaDB with metadata for semantic search
    """
    try:
        config_path = Path(f"documents/onboarding/projects/{project_name}/merged_config.json")
       
        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Config not found. Run merge for project {project_name} first."
            )
       
        print(f"\n{'='*60}")
        print(f"Indexing project: {project_name}")
        print(f"{'='*60}")
       
        # Use embedder to chunk and embed
        result = embedder.embed_project(project_name)
       
        print(f"✓ Successfully indexed {result['chunks_embedded']} chunks")
        print(f"{'='*60}\n")
       
        return {
            "success": True,
            "message": f"Indexed {result['chunks_embedded']} chunks for {project_name}",
            "chunks_embedded": result['chunks_embedded'],
            "chunk_ids": result['chunk_ids'],
            "embedding_model": result.get('embedding_model', 'text-embedding-3-small')
        }
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")
 
 
@app.post("/query", response_model=QueryResponse)
async def query_onboarding(query: QueryRequest):
    """
    Query onboarding information using RAG with LangChain
    """
    try:
        print(f"\n{'='*60}")
        print(f"RAG Query with LangChain")
        print(f"{'='*60}")
        print(f"Question: {query.question[:100]}...")
       
        project_id = query.project
        role = query.role
       
        print(f"Using: project={project_id}, role={role}")
       
        if not llm:
            raise HTTPException(
                status_code=503,
                detail="OpenAI is not configured. Please set OPENAI_API_KEY in .env file"
            )
       
        # Search documents (embedder.query only supports project_id and phase, not role)
        search_results = embedder.query(
            query_text=query.question,
            project_id=project_id,
            n_results=3
        )
       
        # Build context from search results
        context_parts = []
        for doc in search_results['documents']:
            context_parts.append(doc)
        context = "\n\n".join(context_parts)
       
        # Build LangChain messages
        system_msg = SystemMessage(content="""You are an expert employee onboarding assistant.
Use the provided context to answer questions clearly and accurately.
Provide actionable steps, timelines, and specific resources.""")
       
        user_msg = HumanMessage(content=f"""Context from documentation:
{context}
 
Question: {query.question}
 
Please provide a clear, helpful answer based on the context.""")
       
        # Call OpenAI via LangChain
        print("🤖 Calling OpenAI via LangChain...")
        response = call_openai_with_retry([system_msg, user_msg])
       
        answer = response.content
       
        # Extract sources from search results
        sources = search_results.get('metadatas', [])
       
        print(f"✓ Response generated: {len(answer)} characters")
        print(f"✓ Sources: {len(sources)}")
        print(f"{'='*60}\n")
       
        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata={
                "model": llm.model_name,
                "tokens_used": 0,  # LangChain doesn't expose token count easily
                "project": project_id,
                "role": role
            }
        )
   
    except Exception as e:
        print(f"✗ Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
 
 
def parse_user_query(question: str) -> Dict[str, str]:
    """
    Parse user query to extract role, project, and region information
   
    Example: "I am a Senior Backend Developer (Java Spring Boot), I will join AC1, region EU"
    """
    parsed = {}
   
    # Extract project (AC1, EU-BankX, etc.)
    project_match = re.search(r'(?:join|joining|project)\s+([A-Z][A-Z0-9\-]+)', question, re.IGNORECASE)
    if project_match:
        parsed['project'] = project_match.group(1)
   
    # Extract region
    region_match = re.search(r'region\s+(EU|US|APAC|Europe|America|Asia)', question, re.IGNORECASE)
    if region_match:
        region = region_match.group(1).upper()
        if region in ['EUROPE']:
            region = 'EU'
        elif region in ['AMERICA']:
            region = 'US'
        elif region in ['ASIA']:
            region = 'APAC'
        parsed['region'] = region
   
    # Extract role
    role_keywords = {
        'backend': ['backend', 'back-end', 'server-side', 'api', 'spring boot', 'java', 'python', 'node'],
        'frontend': ['frontend', 'front-end', 'react', 'vue', 'angular', 'ui', 'ux'],
        'fullstack': ['fullstack', 'full-stack', 'full stack'],
        'devops': ['devops', 'dev-ops', 'infrastructure', 'kubernetes', 'docker', 'ci/cd'],
        'qa': ['qa', 'quality assurance', 'tester', 'test engineer', 'automation'],
        'data': ['data engineer', 'data scientist', 'ml engineer', 'machine learning']
    }
   
    question_lower = question.lower()
    for role_type, keywords in role_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            parsed['role'] = role_type
            break
   
    # Extract seniority
    if 'senior' in question_lower or 'sr' in question_lower:
        parsed['seniority'] = 'senior'
    elif 'junior' in question_lower or 'jr' in question_lower:
        parsed['seniority'] = 'junior'
    elif 'lead' in question_lower or 'principal' in question_lower:
        parsed['seniority'] = 'lead'
   
    return parsed
 
 
@app.get("/onboarding-status/{employee_id}")
async def get_onboarding_status(employee_id: str):
    """
    Get onboarding status for an employee
    (This is a placeholder - implement with actual database)
    """
    # TODO: Implement actual status tracking
    return {
        "employee_id": employee_id,
        "message": "Status tracking not yet implemented",
        "note": "Connect to your employee database"
    }
 
 
@app.post("/onboarding-status/{employee_id}/update")
async def update_onboarding_status(employee_id: str, status: OnboardingStatus):
    """
    Update onboarding status for an employee
    (This is a placeholder - implement with actual database)
    """
    # TODO: Implement actual status tracking
    return {
        "success": True,
        "message": "Status tracking not yet implemented",
        "note": "Connect to your employee database"
    }
 
 
# ===========================
# Text-to-Speech Endpoints
# ===========================
 
from fastapi.responses import FileResponse
from tts_service import get_tts_service
 
class TTSRequest(BaseModel):
    """Text-to-Speech request"""
    text: str = Field(..., description="Text to convert to speech")
    engine: Optional[str] = Field(None, description="TTS engine: google, system")
 
class TTSResponse(BaseModel):
    """Text-to-Speech response"""
    success: bool
    audio_url: Optional[str] = None
    engine_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
 
@app.post("/api/text-to-speech", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech and return audio file
   
    Supports multiple TTS engines:
    - google: Google TTS (gTTS) - Free, requires internet (default)
    - system: System TTS (pyttsx3) - Offline, platform-specific
    """
    try:
        # Get TTS service with specified engine
        tts = get_tts_service(engine_type=request.engine)
       
        if not tts.is_available():
            return TTSResponse(
                success=False,
                error=f"TTS engine '{request.engine or 'default'}' is not available. Check installation and configuration."
            )
       
        # Generate speech
        audio_path = tts.text_to_speech(request.text)
       
        if audio_path:
            # Return URL to audio file (will be served by /api/audio endpoint)
            audio_filename = Path(audio_path).name
            return TTSResponse(
                success=True,
                audio_url=f"/api/audio/{audio_filename}",
                engine_info=tts.get_engine_info()
            )
        else:
            return TTSResponse(
                success=False,
                error="Failed to generate speech audio"
            )
   
    except Exception as e:
        return TTSResponse(
            success=False,
            error=f"TTS error: {str(e)}"
        )
 
 
@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serve audio file generated by TTS
    """
    import tempfile
    audio_path = Path(tempfile.gettempdir()) / filename
   
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
   
    # Determine media type based on extension
    media_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"
   
    return FileResponse(
        path=audio_path,
        media_type=media_type,
        filename=filename
    )
 
 
@app.get("/api/tts-info")
async def get_tts_info():
    """
    Get information about available TTS engines (free only)
    """
    tts = get_tts_service()
    return {
        "current_engine": tts.get_engine_info(),
        "available_engines": {
            "google": "Google TTS (gTTS) - Free, requires internet (default)",
            "system": "System TTS (pyttsx3) - Offline, platform-specific"
        },
        "configuration": {
            "TTS_ENGINE": os.getenv("TTS_ENGINE", "google"),
            "TTS_LANGUAGE": os.getenv("TTS_LANGUAGE", "en")
        }
    }
 
 
if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
 
 