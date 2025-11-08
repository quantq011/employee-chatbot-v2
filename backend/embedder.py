"""
Chunking and Embedding Module for Onboarding Documents
Handles intelligent chunking of merged configs and embedding into ChromaDB
Uses OpenAI embeddings via LangChain
"""
import json
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
 
# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")
 
 
class LangChainEmbeddingFunction:
    """Custom embedding function using LangChain OpenAI embeddings"""
   
    def __init__(self):
        """Initialize OpenAI embeddings via LangChain"""
        # Use separate embedding API key if provided, otherwise fall back to main key
        embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        embedding_api_base = os.getenv("OPENAI_EMBEDDING_API_BASE") or os.getenv("OPENAI_API_BASE", "https://aiportalapi.stu-platform.live/jpe")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
       
        # Ensure endpoint ends with /v1 for OpenAI compatibility
        if not embedding_api_base.endswith('/v1'):
            embedding_api_base = embedding_api_base.rstrip('/') + '/v1'
       
        # Create HTTP client with timeout and SSL verification disabled for custom endpoints
        # http_client = httpx.Client(
        #     timeout=httpx.Timeout(60.0, connect=10.0),
        #     limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        #     verify=False  # Disabled SSL verification for custom endpoints
        # )
       
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=embedding_api_key,
            base_url=embedding_api_base,
            # http_client=http_client
        )
        print(f"✓ Embedding function initialized")
        print(f"  Model: {embedding_model}")
        print(f"  Endpoint: {embedding_api_base}")
        print(f"  SSL Verification: Disabled")
        print(f"  Timeout: 60s")
   
    def name(self) -> str:
        """Return the name of the embedding function"""
        return "openai_langchain_embeddings"
   
    def embed_query(self, input) -> List[float]:
        """Embed a single query string"""
        try:
            if isinstance(input, list):
                input_text = input[0] if input else ""
            else:
                input_text = input
           
            input_text = input_text.strip()
            if not input_text:
                raise ValueError("Cannot embed empty string")
           
            print(f"Embedding query: {input_text[:100]}...")
            result = self.embeddings.embed_query(input_text)
            print(f"✓ Query embedded: {len(result)} dimensions")
            return result
        except Exception as e:
            print(f"✗ Error embedding query: {e}")
            raise
   
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts using LangChain"""
        if not input:
            return []
       
        try:
            print(f"Embedding {len(input)} documents...")
           
            # Validate and clean inputs
            cleaned_input = []
            for text in input:
                text = text.strip()
                if not text:
                    raise ValueError("Cannot embed empty string")
                cleaned_input.append(text)
           
            # Use LangChain's embed_documents method (handles batching automatically)
            embeddings = self.embeddings.embed_documents(cleaned_input)
           
            print(f"✓ Embedded {len(embeddings)} documents")
            return embeddings
        except Exception as e:
            import traceback
            print(f"✗ Error embedding batch: {type(e).__name__}: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            raise
 
 
class OnboardingEmbedder:
    """Handles chunking and embedding of onboarding documents"""
   
    def __init__(self, chroma_persist_dir: str = "./chroma_db"):
        """Initialize ChromaDB client and collection with OpenAI embeddings"""
        self.client = chromadb.Client(Settings(
            persist_directory=chroma_persist_dir,
            anonymized_telemetry=False
        ))
       
        # Initialize LangChain OpenAI embedding function
        self.embedding_function = LangChainEmbeddingFunction()
       
        self.collection = self.client.get_or_create_collection(
            name="onboarding_chunks",
            metadata={"description": "Chunked onboarding documents with Azure OpenAI embeddings"},
            embedding_function=self.embedding_function
        )
   
    def chunk_merged_config(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Intelligently chunk the merged config into semantic pieces
       
        Returns list of chunks with content and metadata
        """
        chunks = []
        metadata = config.get('metadata', {})
        overrides = config.get('overrides', {})
       
        project_id = metadata.get('project_id', 'unknown')
        region = metadata.get('region', 'unknown')
        version = metadata.get('version', 'unknown')
       
        # Chunk 1: Role information
        if 'role' in overrides:
            role_data = overrides['role']
            role_text = self._format_role_chunk(role_data)
            chunks.append({
                "content": role_text,
                "metadata": {
                    "project_id": project_id,
                    "region": region,
                    "version": version,
                    "type": "role",
                    "role": role_data.get('role', 'Unknown'),
                    "chunk_type": "role_description"
                }
            })
       
        # Chunk 2: Region information
        if 'region' in overrides:
            region_data = overrides['region']
            region_text = self._format_region_chunk(region_data)
            chunks.append({
                "content": region_text,
                "metadata": {
                    "project_id": project_id,
                    "region": region,
                    "version": version,
                    "type": "region",
                    "chunk_type": "region_info"
                }
            })
       
        # Chunk 3-6: Each phase as separate chunk
        if 'phases' in overrides:
            for phase_name, phase_data in overrides['phases'].items():
                phase_text = self._format_phase_chunk(phase_name, phase_data)
                chunks.append({
                    "content": phase_text,
                    "metadata": {
                        "project_id": project_id,
                        "region": region,
                        "version": version,
                        "type": "phase",
                        "phase": phase_name,
                        "chunk_type": "onboarding_phase"
                    }
                })
       
        # Chunk 7: Project-specific information
        if 'project_specific' in overrides:
            project_specific = overrides['project_specific']
            project_text = self._format_project_specific_chunk(project_specific)
            chunks.append({
                "content": project_text,
                "metadata": {
                    "project_id": project_id,
                    "region": region,
                    "version": version,
                    "type": "project_specific",
                    "chunk_type": "project_details"
                }
            })
       
        return chunks
   
    def _format_role_chunk(self, role_data: Dict[str, Any]) -> str:
        """Format role data into readable text"""
        lines = [
            f"# Role: {role_data.get('role', 'Unknown')}",
            "",
            f"Description: {role_data.get('description', 'N/A')}",
            ""
        ]
       
        if 'responsibilities' in role_data:
            lines.append("## Responsibilities:")
            for resp in role_data['responsibilities']:
                lines.append(f"- {resp}")
            lines.append("")
       
        if 'required_skills' in role_data:
            lines.append("## Required Skills:")
            for skill in role_data['required_skills']:
                lines.append(f"- {skill}")
            lines.append("")
       
        if 'tools' in role_data:
            lines.append("## Tools and Technologies:")
            lines.append(", ".join(role_data['tools']))
            lines.append("")
       
        if 'onboarding_tasks' in role_data:
            lines.append("## Onboarding Tasks:")
            for task in role_data['onboarding_tasks']:
                lines.append(f"- {task}")
            lines.append("")
       
        if 'additional_responsibilities' in role_data:
            lines.append("## Additional Responsibilities:")
            for resp in role_data['additional_responsibilities']:
                lines.append(f"- {resp}")
            lines.append("")
       
        return "\n".join(lines)
   
    def _format_region_chunk(self, region_data: Dict[str, Any]) -> str:
        """Format region data into readable text"""
        lines = [
            f"# Region: {region_data.get('region', 'Unknown')}",
            "",
            f"Timezone: {region_data.get('timezone', 'N/A')}",
            f"Work Hours: {region_data.get('work_hours', 'N/A')}",
            ""
        ]
       
        if 'culture' in region_data:
            culture = region_data['culture']
            lines.append("## Cultural Information:")
            lines.append(f"- Meeting Style: {culture.get('meeting_style', 'N/A')}")
            lines.append(f"- Communication: {culture.get('communication', 'N/A')}")
            lines.append(f"- Work-Life Balance: {culture.get('work_life_balance', 'N/A')}")
            lines.append("")
       
        if 'compliance' in region_data:
            compliance = region_data['compliance']
            lines.append("## Compliance Requirements:")
            for key, value in compliance.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
       
        if 'local_contacts' in region_data:
            lines.append("## Local Contacts:")
            for role, contact in region_data['local_contacts'].items():
                lines.append(f"- {role}: {contact}")
            lines.append("")
       
        return "\n".join(lines)
   
    def _format_phase_chunk(self, phase_name: str, phase_data: Dict[str, Any]) -> str:
        """Format phase data into readable text"""
        lines = [
            f"# Onboarding Phase: {phase_data.get('phase', phase_name)}",
            "",
            f"Duration: {phase_data.get('duration', 'N/A')}",
            f"Description: {phase_data.get('description', 'N/A')}",
            ""
        ]
       
        if 'objectives' in phase_data:
            lines.append("## Objectives:")
            for obj in phase_data['objectives']:
                lines.append(f"- {obj}")
            lines.append("")
       
        if 'daily_breakdown' in phase_data:
            lines.append("## Daily Breakdown:")
            for day, tasks in phase_data['daily_breakdown'].items():
                lines.append(f"\n### {day.replace('_', ' ').title()}:")
                for task in tasks:
                    lines.append(f"- {task}")
            lines.append("")
       
        if 'activities' in phase_data:
            lines.append("## Activities:")
            for activity in phase_data['activities']:
                lines.append(f"- {activity}")
            lines.append("")
       
        if 'technical_tasks' in phase_data:
            lines.append("## Technical Tasks:")
            for task in phase_data['technical_tasks']:
                lines.append(f"- {task}")
            lines.append("")
       
        if 'focus_areas' in phase_data:
            lines.append("## Focus Areas:")
            for area, items in phase_data['focus_areas'].items():
                lines.append(f"\n### {area.title()}:")
                for item in items:
                    lines.append(f"- {item}")
            lines.append("")
       
        if 'responsibilities' in phase_data:
            lines.append("## Responsibilities:")
            for resp_type, items in phase_data['responsibilities'].items():
                lines.append(f"\n### {resp_type.title()}:")
                for item in items:
                    lines.append(f"- {item}")
            lines.append("")
       
        if 'deliverables' in phase_data:
            lines.append("## Deliverables:")
            for deliverable in phase_data['deliverables']:
                lines.append(f"- {deliverable}")
            lines.append("")
       
        if 'checklist' in phase_data:
            lines.append("## Checklist:")
            for item in phase_data['checklist']:
                lines.append(f"☐ {item}")
            lines.append("")
       
        if 'milestone' in phase_data:
            lines.append(f"## Milestone:")
            lines.append(phase_data['milestone'])
            lines.append("")
       
        # Additional activities from overrides
        if 'additional_activities' in phase_data:
            lines.append("## Additional Activities (Project-Specific):")
            for activity in phase_data['additional_activities']:
                lines.append(f"- {activity}")
            lines.append("")
       
        if 'additional_tasks' in phase_data:
            lines.append("## Additional Tasks (Project-Specific):")
            for task in phase_data['additional_tasks']:
                lines.append(f"- {task}")
            lines.append("")
       
        return "\n".join(lines)
   
    def _format_project_specific_chunk(self, project_data: Dict[str, Any]) -> str:
        """Format project-specific data into readable text"""
        lines = ["# Project-Specific Information", ""]
       
        if 'repositories' in project_data:
            lines.append("## Repositories:")
            for repo in project_data['repositories']:
                lines.append(f"- {repo}")
            lines.append("")
       
        if 'slack_channels' in project_data:
            lines.append("## Slack Channels:")
            for channel in project_data['slack_channels']:
                lines.append(f"- {channel}")
            lines.append("")
       
        if 'contacts' in project_data:
            lines.append("## Key Contacts:")
            for role, contact in project_data['contacts'].items():
                lines.append(f"- {role}: {contact}")
            lines.append("")
       
        if 'special_requirements' in project_data:
            lines.append("## Special Requirements:")
            for req, value in project_data['special_requirements'].items():
                lines.append(f"- {req}: {value}")
            lines.append("")
       
        return "\n".join(lines)
   
    def embed_project(self, project_name: str, config_path: str = None) -> Dict[str, Any]:
        """
        Load merged config, chunk it, and embed into ChromaDB
       
        Args:
            project_name: Name of the project
            config_path: Optional custom path to merged config
       
        Returns:
            Dictionary with embedding results
        """
        # Load merged config
        if config_path:
            path = Path(config_path)
        else:
            path = Path(f"documents/onboarding/projects/{project_name}/merged_config.json")
       
        if not path.exists():
            raise FileNotFoundError(f"Merged config not found: {path}")
       
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
       
        # Generate chunks
        chunks = self.chunk_merged_config(config)
       
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
       
        project_id = config.get('metadata', {}).get('project_id', project_name)
       
        for i, chunk in enumerate(chunks):
            doc_id = f"{project_id}_{chunk['metadata']['type']}_{i}"
            if chunk['metadata']['type'] == 'phase':
                doc_id = f"{project_id}_phase_{chunk['metadata']['phase']}"
           
            documents.append(chunk['content'])
            metadatas.append(chunk['metadata'])
            ids.append(doc_id)
       
        # Add to ChromaDB
        print(f"Adding {len(documents)} chunks to ChromaDB with Azure OpenAI embeddings...")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✓ Successfully embedded {len(chunks)} chunks for project '{project_name}'")
       
        return {
            "success": True,
            "project": project_name,
            "chunks_embedded": len(chunks),
            "chunk_ids": ids,
            "embedding_model": "text-embedding-3-small"
        }
   
    def query(self, query_text: str, project_id: str = None,
              phase: str = None, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the embedded documents
       
        Args:
            query_text: The query string
            project_id: Optional project filter
            phase: Optional phase filter
            n_results: Number of results to return
       
        Returns:
            Query results with documents and metadata
        """
        # Manually embed the query using our embedding function
        print(f"Generating query embedding...")
        query_embedding = self.embedding_function([query_text])[0]  # Get first embedding
        print(f"✓ Query embedding generated: {len(query_embedding)} dimensions")
       
        # Build where filter with proper ChromaDB syntax
        where_filter = None
        conditions = []
       
        if project_id:
            conditions.append({"project_id": project_id})
        if phase:
            conditions.append({"phase": phase})
       
        # ChromaDB requires $and operator when multiple conditions
        if len(conditions) > 1:
            where_filter = {"$and": conditions}
        elif len(conditions) == 1:
            where_filter = conditions[0]
       
        # Use query_embeddings instead of query_texts
        results = self.collection.query(
            query_embeddings=[query_embedding],  # Pass the embedding directly
            n_results=n_results,
            where=where_filter
        )
       
        return {
            "documents": results['documents'][0] if results['documents'] else [],
            "metadatas": results['metadatas'][0] if results['metadatas'] else [],
            "distances": results['distances'][0] if results.get('distances') else []
        }
 
 
if __name__ == "__main__":
    # Test embedding
    embedder = OnboardingEmbedder()
   
    # Embed AC1 project
    result = embedder.embed_project("AC1")
    print(json.dumps(result, indent=2))
   
    # Test query
    query_result = embedder.query(
        "What should I do in the first week?",
        project_id="AC1"
    )
   
    print("\n=== Query Results ===")
    for i, doc in enumerate(query_result['documents'][:2]):
        print(f"\n--- Result {i+1} ---")
        print(doc[:500])
 
 