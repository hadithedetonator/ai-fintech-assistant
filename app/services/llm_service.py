import logging
import os
import glob
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
from app.services.prompts import ANALYST_PROMPTS

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.keys = {
            "openai": OPENAI_API_KEY,
            "gemini": GEMINI_API_KEY,
            "deepseek": DEEPSEEK_API_KEY
        }
        self.cached_models = {}
        self.cached_embeddings = {}
        logger.info(f"LLM Service initialized with available providers: {[p for p, k in self.keys.items() if k]}")

    def _get_llm(self, provider: str):
        """Lazy initialization of LLM based on provider with caching."""
        if provider in self.cached_models:
            return self.cached_models[provider]
            
        llm = None
        if provider == "openai" and self.keys["openai"]:
            llm = ChatOpenAI(api_key=self.keys["openai"], model="gpt-4o", temperature=0)
        
        elif provider == "gemini" and self.keys["gemini"]:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.keys["gemini"], temperature=0)
        
        elif provider == "deepseek" and self.keys["deepseek"]:
            llm = ChatOpenAI(
                api_key=self.keys["deepseek"],
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                temperature=0
            )

        elif provider == "local":
            llm = self._get_local_llm()
            
        if llm:
            self.cached_models[provider] = llm
        return llm

    def _get_local_llm(self):
        """Attempt to load a model from app/models (GGUF or Safetensors)."""
        import os
        import glob
        from pathlib import Path
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        # Get absolute path to the 'models' directory
        try:
            # Robust path resolution
            current_file_path = Path(__file__).resolve()
            app_dir = current_file_path.parent.parent
            model_dir = app_dir / "models"
            
            logger.info(f"DEBUG: Current file: {current_file_path}")
            logger.info(f"DEBUG: App dir: {app_dir}")
            logger.info(f"DEBUG: Checking for local models in: {model_dir}")
            
            if not model_dir.exists():
                logger.error(f"❌ Model directory does not exist: {model_dir}")
                return None
                
            # List files for debugging
            files = [f.name for f in model_dir.iterdir()]
            logger.info(f"DEBUG: Files in model dir: {files}")

        except Exception as e:
            logger.error(f"❌ Error resolving model path: {e}")
            return None
        
        # Check for Safetensors (Transformers)
        # We need BOTH the model file and config files for this to work
        safetensor_path = model_dir / "model.safetensors"
        config_path = model_dir / "config.json"
        
        if safetensor_path.exists():
            if not config_path.exists():
                logger.warning(f"⚠️ Found {safetensor_path} but missing config.json. Transformers might fail.")
                
            logger.info(f"🚀 Found {safetensor_path}. Attempting to load using Transformers...")
            try:
                model_str_path = str(model_dir) # Transformers expects a string path
                
                # Determine device
                device = "cpu"
                if torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("⚡ Using MPS (Metal Performance Shaders) acceleration")
                elif torch.cuda.is_available():
                    device = "cuda"
                    logger.info("⚡ Using CUDA acceleration")
                else:
                    logger.info("🐢 Using CPU (no acceleration detected)")

                logger.info("⏳ Loading Tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_str_path, trust_remote_code=True)
                
                logger.info(f"⏳ Loading Model on {device}...")
                # Try loading directly with pipeline first for simplicity, 
                # or fallback to AutoModel if precise control needed.
                # Using pipeline(model=path) is the user's suggestion and often more robust.
                
                pipe = pipeline(
                    "text-generation",
                    model=model_str_path,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    device=device, # Explicit device handling
                    trust_remote_code=True
                )
                
                logger.info("✅ Model loaded successfully!")
                return HuggingFacePipeline(pipeline=pipe)
                
            except Exception as e:
                err_msg = f"❌ Failed to load safetensors model: {str(e)}"
                print(f"ERROR: {err_msg}")
                logger.error(err_msg, exc_info=True)
                # Fall through to GGUF check

        # Check for GGUF (LlamaCpp)
        gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
        if gguf_files:
            selected_model = gguf_files[0]
            logger.info(f"🚀 Initializing Local GGUF Model: {os.path.basename(selected_model)}")
            try:
                return LlamaCpp(
                    model_path=selected_model,
                    temperature=0,
                    max_tokens=2000,
                    n_ctx=4096,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Failed to load GGUF model: {e}")
                
        return None

    def _get_embeddings(self):
        """Get best available embedding model."""
        if self.keys["openai"]:
            return OpenAIEmbeddings(api_key=self.keys["openai"])
        if self.keys["gemini"]:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(google_api_key=self.keys["gemini"], model="models/gemini-embedding-001")
        
        # Absolute last resort: Local HuggingFace Embeddings
        try:
            logger.info("Initializing Local HuggingFace Embeddings (all-MiniLM-L6-v2)")
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load local embeddings: {e}")
            return None

    async def get_chat_response(self, query: str, transactions: List[Dict[str, Any]], analyst_type: str = "advisor"):
        # 1. Prepare Documents
        docs = []
        for tx in transactions:
            content = f"Date: {tx.get('date', 'N/A')} | Amount: {tx.get('amount', 0)} {tx.get('currency', 'USD')} | Description: {tx.get('description', 'N/A')} | Category: {tx.get('category', 'N/A')}"
            docs.append(Document(page_content=content, metadata=tx))

        # 2. Try Providers in Order: OpenAI -> DeepSeek -> Gemini -> Local
        providers = ["openai", "deepseek", "gemini", "local"]
        errors = []
        tried_status = []

        system_prompt_base = ANALYST_PROMPTS.get(analyst_type, ANALYST_PROMPTS["advisor"])
        full_system_prompt = (
            f"{system_prompt_base}\n\n"
            "Use `retrieve_financial_context` to get transaction data. Be concise and professional."
        )

        for provider in providers:
            try:
                # a) Check if LLM key exists
                llm = self._get_llm(provider)
                if not llm:
                    if provider == "local":
                        # Be more specific for local check
                        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
                        has_safetensors = os.path.exists(os.path.join(model_dir, "model.safetensors"))
                        has_gguf = len(glob.glob(os.path.join(model_dir, "*.gguf"))) > 0
                        
                        status = ""
                        if not (has_safetensors or has_gguf):
                            status = "(Skipped: No Model File Found)"
                        else:
                            status = "(Skipped: Found file but failed to load. Check logs.)"
                        
                        msg = f"⏭️ {provider.upper()} {status}"
                        tried_status.append(msg)
                        logger.info(msg) # Log skip for visibility
                    else:
                        msg = f"⏭️ {provider.upper()} (Skipped: No Key)"
                        tried_status.append(msg)
                        logger.info(msg) # Log skip for visibility
                    continue

                # b) Check if we have embeddings for this provider's flow
                # For RAG, we need to create the tool WITH a working vector store
                msg = f"🔄 [EXECUTING] Attempting RAG with: {provider.upper()}"
                print(f"\n{msg}")
                logger.info(msg)
                tried_status.append(f"🔄 {provider.upper()} (Attempted)")

                # Try to get embeddings for this path
                emb_model = None
                if provider == "openai" and self.keys["openai"]:
                    if "openai" not in self.cached_embeddings:
                        self.cached_embeddings["openai"] = OpenAIEmbeddings(api_key=self.keys["openai"])
                    emb_model = self.cached_embeddings["openai"]
                    
                elif provider == "gemini" and self.keys["gemini"]:
                    if "gemini" not in self.cached_embeddings:
                        from langchain_google_genai import GoogleGenerativeAIEmbeddings
                        self.cached_embeddings["gemini"] = GoogleGenerativeAIEmbeddings(google_api_key=self.keys["gemini"], model="models/gemini-embedding-001")
                    emb_model = self.cached_embeddings["gemini"]
                    
                else:
                    # Fallback to local embeddings
                    if "local" not in self.cached_embeddings:
                        logger.info("Initializing Local HuggingFace Embeddings for this attempt.")
                        self.cached_embeddings["local"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    emb_model = self.cached_embeddings["local"]

                # c) Build Vector Store (This triggers API calls for API-based embeddings)
                vector_store = InMemoryVectorStore(emb_model)
                vector_store.add_documents(docs)

                # d) Define Retrieval Tool
                @tool(response_format="content_and_artifact")
                def retrieve_financial_context(search_query: str):
                    """Retrieve relevant transactional information to help answer a financial query."""
                    retrieved_docs = vector_store.similarity_search(search_query, k=10)
                    serialized = "\n\n".join(f"Transaction: {doc.page_content}" for doc in retrieved_docs)
                    return serialized, retrieved_docs

                # e) Run Agent or Direct RAG
                if provider == "local":
                    # Local models often don't support tool binding (.bind_tools)
                    # We perform a direct RAG flow
                    msg = "🏠 [LOCAL RAG] Fetching context manually..."
                    print(msg)
                    logger.info(msg)
                    
                    # Tool invoke might return just content or (content, artifact)
                    # depending on LangChain version and tool configuration.
                    tool_output = retrieve_financial_context.invoke(query)
                    if isinstance(tool_output, tuple) and len(tool_output) >= 1:
                        context_text = tool_output[0]
                    else:
                        context_text = str(tool_output) # Ensure string
                        
                    local_prompt = f"{full_system_prompt}\n\nRELEVANT CONTEXT:\n{context_text}\n\nUSER QUERY: {query}"
                    
                    # HuggingFacePipeline is an LLM, not a ChatModel, so we use invoke directly
                    answer = llm.invoke(local_prompt)
                    
                    success_msg = f"✅ [SUCCESS] Request fulfilled by {provider.upper()}!"
                    print(success_msg)
                    logger.info(success_msg)
                    return answer
                else:
                    # API models usually support tool binding via langchain's create_agent
                    agent = create_agent(llm, [retrieve_financial_context], system_prompt=full_system_prompt)
                    response = await agent.ainvoke({"messages": [("user", query)]})
                    
                    if "messages" in response:
                        answer = response["messages"][-1].content
                        success_msg = f"✅ [SUCCESS] Request fulfilled by {provider.upper()}!"
                        print(success_msg)
                        logger.info(success_msg)
                        return answer
                    
                    return str(response)

            except Exception as e:
                err_label = f"❌ {provider.upper()} FAILED"
                print(f"{err_label}: {str(e)}")
                logger.warning(f"{err_label}: {str(e)}")
                errors.append(f"{err_label}: {str(e)}")
                # Continue loop to next provider
                continue

        # If all failed
        trace = " -> ".join(tried_status)
        combined_errors = "\n\n".join(errors)
        
        raise RuntimeError(
            f"All AI providers failed or were unavailable.\n\n"
            f"EXECUTION TRACE: {trace}\n\n"
            f"DETAILED ERRORS:\n{combined_errors}\n\n"
            f"Please check your API balances or add a .gguf model to app/models/."
        )
