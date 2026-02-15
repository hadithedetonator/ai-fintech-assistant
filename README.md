A production-grade AI service for financial insights using **FastAPI**, **LangChain Agentic RAG**, and **Multi-Model Support (OpenAI, Gemini, DeepSeek)**.

## 🚀 Key Features

*   **Agentic RAG**: Unlike simple prompt injection, this service uses a LangChain Agent equipped with a `retrieve_financial_context` tool. It performs semantic search over the transactions you provide in the request.
*   **Multi-Model Intelligence**: Automatically falls back through available keys:
    1.  **OpenAI** (gpt-4o)
    2.  **Gemini** (gemini-2.0-flash)
    3.  **DeepSeek** (deepseek-chat)
    4.  **Local Model** (LlamaCpp - absolute fallback if offline or keys fail)

## 🏠 Local Fallback Setup
If you want to run the service completely offline or as a last-resort fallback:
1.  Download a `.gguf` model (e.g., Llama-3-8B-Instruct).
2.  Place it in the `ai-service/app/models/` directory.
3.  The system will automatically detect and use it if all API keys are exhausted.
*   **Header-Based Analyst Personas**: Use the `X-Analyst-Type` header to switch between different AI personalities:
    *   `advisor`: Standard financial advisor.
    *   `tax`: Professional Tax Consultant.
    *   `coach`: Strict Budget Coach.
*   **API Security & Anti-Abuse**:
    *   **Rate Limiting**: Powered by `slowapi` to prevent brute-force or massive usage (current limits: 10 calls/min for chat, 5 calls/min for health).
    *   **Pydantic V2**: Strict validation of incoming financial data.
*   **Scalable Architecture**: Uses an `InMemoryVectorStore` for each request, ensuring zero data leakage between users while providing high-quality semantic retrieval.

---

## 🛠 Setup & Installation

### 1. Environment
```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Service
```bash
uvicorn app.main:app --reload
```

---

## 🛰 API Documentation

### POST `/chat`
**Headers:**
*   `Content-Type: application/json`
*   `X-Analyst-Type`: (Optional) `advisor`, `tax`, or `coach`.

**Request Body:**
```json
{
  "msg": "Am I spending too much on entertainment?",
  "transactions": [
    {
      "id": "1",
      "amount": 15.0,
      "currency": "USD",
      "description": "Netflix",
      "date": "2024-02-01",
      "category": "Entertainment"
    }
  ]
}
```

---

## 🧪 Testing & RAG Flow
See [TESTING.md](./TESTING.md) for detailed test cases and architectural diagrams.
