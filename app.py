from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
import threading
import time

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_URL = "https://brainlox.com"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the QA chain and conversation memories
qa_chain = None
# Dictionary to store conversation memories for different users
conversation_memories = {}

# ✅ 1️⃣ Scrape Course Listings
def scrape_course_listing():
    url = "https://brainlox.com/courses/category/technical"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    courses = []
    for course in soup.find_all('div', class_='single-courses-box'):
        title = course.find('h3').text.strip()

        price_tag = course.find('span', class_='price-per-session')
        price_per_session = price_tag.text.strip() if price_tag else "N/A"

        number_of_sessions_tag = course.find('ul', class_='course-features-list')
        number_of_sessions = number_of_sessions_tag.find_all('li')[0].text.strip() if number_of_sessions_tag else "N/A"

        course_url = BASE_URL + course.find('a', class_='d-block image')['href']
        
        courses.append({
            'title': title,
            'price_per_session': price_per_session,
            'number_of_sessions': number_of_sessions,
            'number_of_lessons': number_of_sessions,
            'course_url': course_url
        })

    return courses

# ✅ 2️⃣ Scrape Individual Course Details
def scrape_course_details(course_url):
    response = requests.get(course_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    title_tag = soup.select_one('.page-title-area .container .page-title-content h2')
    title = title_tag.text.strip() if title_tag else "Unknown"

    overview, duration, price = "Not Available", "Not Specified", "N/A"
    details_div = soup.find('div', class_='courses-details-desc')
    
    if details_div:
        overview_div = details_div.find('div', class_='courses-overview')
        if overview_div:
            p_tag = overview_div.find('p')
            if p_tag:
                overview = p_tag.text.strip()

    details_info = soup.find('div', class_='courses-details-info')
    if details_info:
        info_list = details_info.find('ul', class_='info')
        if info_list:
            for li in info_list.find_all('li'):
                if li.find('i', class_='flaticon-time'):
                    duration = li.get_text(strip=True)
                if li.find('i', class_='flaticon-tag'):
                    price_text = li.get_text(strip=True)
                    price = price_text.split(":")[-1].strip()

    return {
        'title': title,
        'overview': overview,
        'duration': duration,
        'total_price': price,
    }

# ✅ 3️⃣ Process All Courses
def process_courses():
    courses = scrape_course_listing()
    for course in courses:
        course_details = scrape_course_details(course['course_url'])
        course.update(course_details)
    return courses

# ✅ 4️⃣ Prepare Documents for Vector Storage
def prepare_documents(courses):
    documents = []
    for i, course in enumerate(courses):
        text = (
            f"**Course Title:** {course['title']}\n"
            f"**Overview:** {course['overview']}\n"
            f"**Duration:** {course['duration']}\n"
            f"**Total Price:** {course['total_price']}\n"
            f"**Price Per Session or Lessons:** {course.get('price_per_session', 'N/A')}\n"
            f"**Number of Sessions or Lessons:** {course.get('number_of_sessions', 'N/A')}\n"
            f"🔗 **Course URL:** {course['course_url']}"
        )
        documents.append({"id": str(i), "text": text, "metadata": course})
    return documents

# ✅ 5️⃣ Create Vector Store
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts, metadatas = [], []

    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": doc["id"], **doc["metadata"]})

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

    return Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas, persist_directory="./chroma_db")

# ✅ 6️⃣ Create RAG Chatbot
def create_rag_chatbot(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )

    # Changed to use MMR retrieval as in the updated code
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})

    # Updated prompt template
    system_template = """You are Brainlox's AI assistant, here to help users find the perfect course for their learning journey. Your goal is to provide clear, friendly, and informative answers about Brainlox courses.

🔹 **Brainlox's Mission:**  
Brainlox is dedicated to bringing significant changes in online learning by researching course curriculums, improving student engagement, and making education more flexible and accessible.

✅ **How You Should Respond:**  
- ✅ **For questions about available courses in a specific domain/field (e.g., "What web development courses do you have?"):**  
  - List all **relevant** courses (course['title']) based on similarity search.  
  - Format as a clean, bulleted list.  
  - **Ensure at least 2-3 courses** are included, if available.  
  - Avoid listing duplicates.  
  - If no exact match is found, suggest similar courses.  
  - **Example Format:**  
    - Web Development Bootcamp  
    - Advanced Frontend Engineering  
    - Full-Stack Web Development  
  - End with:  
    *"Would you like details about any specific course?"*

- **For questions about specific courses:**  
- Provide complete course details, including:
  - **Title:** course['title']
  - **Overview:** course['overview']
  - **Total Price:** course['total_price']  
  - **Price Per Session:** course['price_per_session']
    - If any value is missing, say: "Not specified"  
  - **Duration:** course['duration']

  - If a particular field is missing, state "Not specified" rather than suggesting contacting support
  - ALWAYS include the direct course URL at the end of your response: course['course_url']
  - Format course links as: "Learn more and enroll here: [Course Title](course_url)"
  - Keep responses friendly and engaging

- **For general learning-related questions:**  
  - Offer helpful insights while staying relevant to online learning
  - If a Brainlox course covers the topic, suggest it with its complete URL

- **For unrelated questions:**  
  - Provide a brief answer if possible, then politely mention that you specialize in Brainlox courses

🔗 **For More Help:**  
You can visit the course page for details: [Brainlox Courses](https://brainlox.com/courses)  
For further assistance, reach out at **support@brainlox.com** or call **(+1) 414 429 3937**.  

{context}

Chat History:  
{chat_history}  

User: {question}  
Assistant:
"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=system_template
    )
    
    # Note: We don't include memory here as we'll manage it separately for each user
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        memory=None  # We'll manage memory per user
    )

# Get or create memory for a user
def get_user_memory(user_id):
    if user_id not in conversation_memories:
        conversation_memories[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
    return conversation_memories[user_id]

# Initialize the model once at startup
def initialize_model():
    global qa_chain
    print("Initializing RAG model...")
    courses = process_courses()
    documents = prepare_documents(courses)
    vectorstore = create_vector_store(documents)
    qa_chain = create_rag_chatbot(vectorstore)
    print("RAG model initialized successfully!")

# ✅ API Routes
@app.route('/api/chat', methods=['POST'])
def chat():
    global qa_chain
    
    # Check if model is initialized
    if qa_chain is None:
        return jsonify({"error": "Model not initialized yet. Try again in a moment."}), 503
    
    # Get data from request
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data['message']
    
    # Get user_id from request, default to "default" if not provided
    user_id = data.get('user_id', 'default')
    
    # Get memory for this user
    memory = get_user_memory(user_id)
    
    try:
        # Create a custom chain with user-specific memory
        response = qa_chain.invoke({
            "question": user_message,
            "chat_history": memory.chat_memory.messages
        })
        
        # Save the interaction to memory
        memory.save_context(
            {"question": user_message},
            {"answer": response.get("answer", "")}
        )
        
        # Extract the answer
        answer = response.get("answer", "I'm not sure how to respond to that.")
        
        # Return the response
        return jsonify({
            "response": answer,
            # Optionally return source information
            "sources": [doc.metadata.get("course_url", "") for doc in response.get("source_documents", [])]
        })
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    global qa_chain
    status = "ready" if qa_chain is not None else "initializing"
    return jsonify({"status": status})

@app.route('/api/refresh', methods=['POST'])
def refresh_courses():
    """Endpoint to refresh course data"""
    try:
        def refresh_in_background():
            global qa_chain
            courses = process_courses()
            documents = prepare_documents(courses)
            vectorstore = create_vector_store(documents)
            qa_chain = create_rag_chatbot(vectorstore)
            
        # Start refresh in background
        refresh_thread = threading.Thread(target=refresh_in_background)
        refresh_thread.start()
        
        return jsonify({"status": "refreshing", "message": "Course data refresh started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset-conversation', methods=['POST'])
def reset_conversation():
    """Endpoint to reset a conversation history"""
    data = request.json
    user_id = data.get('user_id', 'default')
    
    if user_id in conversation_memories:
        # Create a new empty memory
        conversation_memories[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
        return jsonify({"status": "success", "message": f"Conversation for user {user_id} has been reset"})
    else:
        return jsonify({"status": "success", "message": f"No conversation found for user {user_id}"})

# Main entry point
if __name__ == '__main__':
    # Initialize the model in a separate thread to not block app startup
    init_thread = threading.Thread(target=initialize_model)
    init_thread.start()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)