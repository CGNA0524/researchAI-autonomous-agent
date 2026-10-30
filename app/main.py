"""
FastAPI Application Entrypoint for the Predixion AI Research Agent.
Provides REST API endpoints for agent orchestration, health checks, and Swagger documentation.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.agent.orchestrator import AgentOrchestrator
from app.config import get_settings
from app.models.request import ResearchRequest
from app.models.response import ResearchResponse
from app.utils.logging import get_logger

logger = get_logger("app.main")
settings = get_settings()

# Global orchestrator instance initialized at startup
orchestrator: AgentOrchestrator = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes application resources, agents, and tool configurations at startup."""
    global orchestrator
    logger.info("Initializing Predixion AI Research Agent service...")
    orchestrator = AgentOrchestrator()
    active_provider = settings.LLM_PROVIDER.lower().strip()
    if active_provider == "openai":
        logger.info(f"Active LLM Provider: OpenAI | Model: {settings.OPENAI_MODEL}")
    else:
        logger.info(f"Active LLM Provider: Groq | Model: {settings.GROQ_MODEL} (Fallback: {settings.GROQ_FALLBACK_MODEL})")
    yield
    logger.info("Shutting down AI Research Agent service...")


app = FastAPI(
    title="Predixion AI Research Agent",
    description=(
        "Production-minded autonomous AI Research Agent. "
        "Performs planning, tool orchestration (web search, webpage extraction), "
        "information filtering, and grounded answer synthesis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for local dashboards, Streamlit, or external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root() -> Dict[str, Any]:
    """Root endpoint returning service identity, version, and documentation link."""
    return {
        "service": "Predixion AI Research Agent",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "research": "/research (POST)"
        }
    }


@app.get("/health", tags=["Monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint reporting service, tool, and LLM readiness."""
    current_settings = get_settings()
    current_llm = getattr(orchestrator, "llm", None) if orchestrator else None
    active_provider = getattr(current_llm, "provider_name", current_settings.LLM_PROVIDER.lower().strip())
    active_model = getattr(current_llm, "active_model", current_settings.OPENAI_MODEL if active_provider == "openai" else current_settings.GROQ_MODEL)

    has_groq_key = bool(current_settings.GROQ_API_KEY and current_settings.GROQ_API_KEY not in ("", "your_groq_api_key_here") and len(current_settings.GROQ_API_KEY) > 5)
    has_openai_key = bool(current_settings.OPENAI_API_KEY and current_settings.OPENAI_API_KEY not in ("", "your_openai_api_key_here") and len(current_settings.OPENAI_API_KEY) > 5)

    return {
        "status": "healthy",
        "environment": current_settings.APP_ENV,
        "llm_provider": active_provider,
        "active_model": active_model,
        "groq_api_configured": has_groq_key,
        "openai_api_configured": has_openai_key,
        "primary_model": current_settings.GROQ_MODEL,
        "fallback_model": current_settings.GROQ_FALLBACK_MODEL,
        "openai_model": current_settings.OPENAI_MODEL,
        "tools_ready": ["web_search", "webpage_extractor"]
    }


@app.post(
    "/research",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Research"]
)
async def conduct_research(request: ResearchRequest) -> ResearchResponse:
    """
    Executes the full agentic research workflow:
    1. Understands question & creates research plan
    2. Selects tools (Web Search & Webpage Extraction)
    3. Gathers, filters, and sanitizes live evidence
    4. Evaluates confidence, assumptions, and limitations
    5. Synthesizes a structured, grounded report with actual URLs
    """
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator()

    try:
        logger.info(f"Received research request: '{request.query}'")
        response = await orchestrator.run(
            user_question=request.query,
            max_search_results=request.max_search_results,
            max_pages_to_extract=request.max_pages_to_extract
        )
        return response

    except ValueError as e:
        logger.error(f"Validation error during research execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.critical(f"Unhandled error in research pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the research workflow: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )
