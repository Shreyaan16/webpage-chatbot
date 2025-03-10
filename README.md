# Brainlox Course Assistant

A Flask-based AI chatbot that answers questions about technical courses from Brainlox based on gemini api key
need to keep the api key inside GEMINI_API_KEY in the .env file

## Features

The Brainlox Course Assistant can:

- ✅ List similar courses when asked about a specific topic
- ✅ Provide detailed information about specific courses including:
  - Course overview
  - Duration
  - Price per session
  - Total price
- ✅ Return "Not specified" when course details are missing
- ✅ Answer general questions while clarifying they are outside its primary scope

## Technical Overview

The chatbot leverages ChromaDB for vector storage of course data, allowing efficient retrieval of relevant information when responding to user queries.

## Setup & Installation

### 1️⃣ Install Dependencies

Install all required libraries using pip:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Flask App

Start the chatbot server with:

```bash
python app.py
```

**Note:** Initial loading takes approximately 1-2 minutes as the chatbot initializes.

### 🩺 Check Chatbot Readiness

Verify the chatbot is operational by running:

```bash
curl http://127.0.0.1:5000/api/health
```

When ready, you should receive:
```json
{"status": "ready"}
```

## 💬 Interacting with the Chatbot

### API Endpoint

The chatbot exposes a single endpoint for interaction:

**URL:** `http://127.0.0.1:5000/api/chat`  
**Method:** POST  
**Content-Type:** application/json

### 📌 Using Postman

1. Open Postman
2. Create a new POST request to `http://127.0.0.1:5000/api/chat`
3. Set the request body to JSON format:
   ```json
   {
     "message": "Your question here"
   }
   ```
4. Click Send to receive the chatbot's response

### 📌 Using cURL

```bash
curl -X POST \
  http://127.0.0.1:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What are the web development courses?"}'
```

## 📜 Example Queries

Here are some sample queries to try:

| Query | Expected Response |
|-------|-------------------|
| "What are the web development courses?" | Lists all web development courses available |
| "Tell me about the Python for AI course." | Provides overview, duration, and pricing for the Python for AI course |
| "How long is the Full-Stack Development course?" | Returns the duration or "Not specified" if not available |
| "Who is the President of the USA?" | Answers but clarifies this is outside its primary scope |

## Troubleshooting

- If the server fails to start, ensure all dependencies are properly installed
- Verify port 5000 is not in use by another application
- Check logs for detailed error messages if the chatbot isn't responding correctly
