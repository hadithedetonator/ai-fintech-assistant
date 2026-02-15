import logging
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="AI Fintech Assistant API",
    description="A production-ready AI service for financial insights using LangChain Agentic RAG.",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Helper to get LLM service
def get_llm_service():
    return LLMService()

@app.api_route("/health", methods=["GET", "HEAD"], tags=["Monitoring"])
@limiter.limit("5/minute")
async def health_check(request: Request):
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "ai-fintech-assistant"}

@app.post("/chat", response_model=ChatResponse, tags=["AI"])
@limiter.limit("10/minute")
async def chat(
    request: Request, 
    payload: ChatRequest, 
    x_analyst_type: str = Header(default="advisor", alias="X-Analyst-Type")
):
    """
    Agentic RAG chat endpoint. 
    Analyst type can be set via 'X-Analyst-Type' header (advisor, tax, coach).
    """
    logger.info(f"Received chat request. Analyst: {x_analyst_type}")
    
    try:
        service = get_llm_service()
        # Use header value or fallback to payload value
        analyst = x_analyst_type if x_analyst_type != "advisor" else payload.analyst_type
        
        # Convert Pydantic models to dict for the LLM service
        transactions_data = [tx.model_dump() for tx in payload.transactions]
        
        response_text = await service.get_chat_response(
            query=payload.msg,
            transactions=transactions_data,
            analyst_type=analyst
        )
        
        return ChatResponse(answer=response_text)
        
    except Exception as e:
        error_msg = f"AI Service Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=error_msg
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"}
    )
