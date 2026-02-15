#!/bin/bash

# Configuration
URL="http://127.0.0.1:8000/chat"
HEADER="X-Analyst-Type: advisor" # Change to 'tax' or 'coach' to test personas

echo "================================================================"
echo "🚀 Starting AI Service Chat & RAG Flow Test"
echo "================================================================"
echo "ℹ️  Test Scenario: Fallback to Local Model Implementation"
echo "    This system follows a specific inference sequence:"
echo "    1. Attempt OpenAI -> 2. Attempt DeepSeek -> 3. Attempt Gemini."
echo "    4. If ALL API keys fail/missing, fallback to LOCAL MODEL."
echo ""
echo "🖥️  GPU/NPU Acceleration Check:"
echo "    - If Local Model is used, it automatically checks for:"
echo "      * NVIDIA GPU (CUDA)"
echo "      * Mac Silicon (MPS)"
echo "    - Watch your server logs for: '⚡ Using [device] acceleration'"
echo "================================================================"
echo "📤 Sending Request to $URL..."

curl -s -X POST "$URL" \
     -H "Content-Type: application/json" \
     -H "$HEADER" \
     -d '{
           "msg": "Can you analyze my February spending? What was my total and what are the top 3 items?",
           "transactions": [
             {"id": "tx1", "amount": 1200.0, "currency": "USD", "description": "Rent Payment", "date": "2024-02-01", "category": "Housing"},
             {"id": "tx2", "amount": 85.20, "currency": "USD", "description": "Whole Foods Market", "date": "2024-02-05", "category": "Groceries"},
             {"id": "tx3", "amount": 15.99, "currency": "USD", "description": "Netflix Subscription", "date": "2024-02-01", "category": "Entertainment"},
             {"id": "tx4", "amount": 45.00, "currency": "USD", "description": "Shell Gas Station", "date": "2024-02-08", "category": "Transport"},
             {"id": "tx5", "amount": 5.50, "currency": "USD", "description": "Starbucks Coffee", "date": "2024-02-10", "category": "Food"},
             {"id": "tx6", "amount": 62.00, "currency": "USD", "description": "Local Bistro", "date": "2024-02-12", "category": "Food"},
             {"id": "tx7", "amount": 120.00, "currency": "USD", "description": "Electricity Bill", "date": "2024-02-15", "category": "Utilities"}
           ]
         }' | python3 -m json.tool

echo -e "\n✅ Test Complete."
