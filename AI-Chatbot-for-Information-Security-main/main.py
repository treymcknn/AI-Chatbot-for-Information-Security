from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import sys
import hashlib
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Groq
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY_HERE")
index_name = "infosec-policies"
index = pc.Index(index_name)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Launching...")

# In-memory cache for saving responses
response_cache = {}

# Retrieve top policy from Pinecone based on similarity
def retrieve_policy(query, top_k=1, similarity_threshold=0.3): # Adjust threshold here (lower = more lenient)
    """Retrieve policy only if similarity score is above threshold."""
    vector = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = index.query(
        namespace="policies",
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )
    if not results["matches"]:
        return None

    # Check if the best match meets the similarity threshold
    best_match = results["matches"][0]
    if best_match["score"] < similarity_threshold:
        return None

    return best_match["metadata"]["text"]

# Generate answer using Groq LLM
def ask_llm(query):
    # Create cache key from query
    cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()

    # Check cache first
    if cache_key in response_cache:
        return response_cache[cache_key]

    # Try to retrieve policy from database
    context = retrieve_policy(query)
    if not context: # Default response if no relevant policy found
        return "Please enter a different question as I can only answer questions based on our company's security policies."

    prompt = ( # Prompt given to the LLM if match found
        f"You are a company IT security professional helping employees understand security policies.\n\n"
        f"INSTRUCTIONS:\n"
        f"- Answer the question directly using only the information provided below\n"
        f"- Rephrase the policy in clear, simple terms\n"
        f"- Do not mention 'the policy says' or 'according to the policy'\n"
        f"- Do not use phrases like 'the provided policy' or 'policy text'\n"
        f"- Just give the information directly as if you're explaining the company rules\n"
        f"- Keep responses to 3 sentences maximum\n"
        f"- If the information doesn't answer the question, just explain what is covered in policy\n\n"
        f"COMPANY SECURITY POLICY:\n{context}\n\n" # Policy match from Pinecone
        f"Question: {query}\n\n" # User question
        f"Answer:" # Answer to be generated
    )

    # Call Groq API
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful cybersecurity policy professional. Provide clear, concise answers based only on the provided policy text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 150,
            "temperature": 0.3,  # Lower temperature for more focused responses
            "top_p": 0.9
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()

        # Cache the response
        response_cache[cache_key] = answer

        return answer

    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}\n\nPlease check the Groq API key and internet connection."
    except Exception as e:
        return f"Error: {str(e)}"

# Runs API call in background to stop GUI freezing
class InferenceThread(QThread):
    result_ready = pyqtSignal(str)

    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self):
        try:
            answer = ask_llm(self.message)
            self.result_ready.emit(answer)
        except Exception as e:
            self.result_ready.emit(f"Error: {str(e)}")

# GUI App
class ChatBotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Chatbot")
        self.setGeometry(100, 100, 500, 600)
        self.layout = QVBoxLayout()

        defualt_font = QFont("Arial", 14)
        self.setFont(defualt_font)

        # Chat display (read-only)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        # Add startup message
        self.chat_display.append("<b>Security Policy Assistant:</b> What security related question can I assist with today?")

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        self.layout.addWidget(self.status_label)

        # User input
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.setFocus()
        self.user_input.returnPressed.connect(self.send_message)
        self.layout.addWidget(self.user_input)

        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.send_button)

        # Exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setStyleSheet("background-color: #dc3545; color: white;")
        self.layout.addWidget(self.exit_button)

        self.setLayout(self.layout)

        # Track if currently processing
        self.is_processing = False
        self.inference_thread = None

    def send_message(self):
        # Prevent multiple simultaneous requests
        if self.is_processing:
            return

        user_text = self.user_input.text().strip()

        if not user_text:
            self.chat_display.append("<b>Bot:</b> Please enter a question")
            self.user_input.clear()
            return

        # Show user message
        self.chat_display.append(f"<b>You:</b> {user_text}")
        self.user_input.clear()

        # Disable input while processing
        self.is_processing = True
        self.user_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.status_label.setText("Processing... (typically 1-3 seconds)")

        # Start inference in background thread
        self.inference_thread = InferenceThread(user_text)
        self.inference_thread.result_ready.connect(self.on_result_ready)
        self.inference_thread.start()

    def on_result_ready(self, bot_reply):
        # Display bot response
        self.chat_display.append(f"<b>Security Policy Assistant:</b> {bot_reply}")

        # Re-enable input
        self.is_processing = False
        self.user_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.status_label.setText("")
        self.user_input.setFocus()

        # Clean up thread
        if self.inference_thread:
            self.inference_thread.quit()
            self.inference_thread.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatBotGUI()
    window.show()
    sys.exit(app.exec_())