# Testing & AI Orchestration (Agentic RAG)

## 🧠 AI Architecture: Agentic RAG

Version 2.0 implements a more sophisticated **Agentic RAG** pattern instead of simple prompt injection.

### The Flow:
1.  **Request Ingestion**: The app receives the `msg` and `transactions`.
2.  **Volatile Indexing**: For the duration of the request, the transactions are converted into `Document` objects and indexed into a LangChain `InMemoryVectorStore`.
3.  **Agent Initialization**: A LangChain agent is created using the persona selected via the `X-Analyst-Type` header.
4.  **Retrieval Step**: The agent is given a tool called `retrieve_financial_context`. 
5.  **Reasoning Loop**: 
    - The Agent analyzes the user query.
    - If it needs data, it calls the retrieval tool.
    - The tool performs a **semantic search** on the vector store.
    - The Agent receives the relevant context and formulates the final answer.

## 🛡 Abuse Prevention (Orchestration)

To ensure the service remains reliable and cost-effective:

1.  **Rate Limiting**: 
    - `/chat`: 10 requests per minute per IP.
    - `/health`: 5 requests per minute per IP.
    - *Exceeding these limits returns a `429 Too Many Requests` status.*
2.  **Context Management**: We limit the retrieval tool to return the top 10 most relevant transactions, preventing prompt bloat.
3.  **Isolation**: Each request creates a new vector store, ensuring no cross-user data interference.

## 🧪 How to Test

### 1. Testing Analyst Flag (Header)
Use Postman or curl to verify the persona shift:

**Tax Analysis:**
```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "X-Analyst-Type: tax" \
     -H "Content-Type: application/json" \
     -d '{"msg": "Do I have any tax deductible donations?", "transactions": [{"id": "1", "amount": 100, "currency": "USD", "description": "UNICEF", "date": "2024-01-01", "category": "Charity"}]}'
```

**Budget Coaching (Strict):**
```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "X-Analyst-Type: coach" \
     -H "Content-Type: application/json" \
     -d '{"msg": "Tell me the truth about my coffee habit.", "transactions": [{"id": "1", "amount": 7, "currency": "USD", "description": "Starbucks", "date": "2024-01-01", "category": "Food"}]}'
```

### 2. Testing Rate Limiting
Try running the health check command 6 times fast:
```bash
for i in {1..6}; do curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/health; done
```
*You should see `200` for the first five calls and `429` on the 6th call.*

### 3. Testing Semantic Search
Ask a question that require context matching, e.g., "Where did I buy electronics?" when the category is "Shopping" but the description is "Best Buy".
