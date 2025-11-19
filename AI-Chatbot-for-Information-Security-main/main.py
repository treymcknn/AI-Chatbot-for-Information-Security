from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import sys
import hashlib
import requests
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# === Groq Configuration ===
# Get your free API key from: https://console.groq.com/keys
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # Replace with your API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# === Pinecone setup ===
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY_HERE")
index_name = "infosec-policies"
index = pc.Index(index_name)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Groq chatbot initialized - ready to go! (Super fast responses)")

# Simple in-memory cache for responses
response_cache = {}

# === Retrieve top policy from Pinecone ===
def retrieve_policy(query, top_k=1):
    vector = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = index.query(
        namespace="policies",
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )
    if not results["matches"]:
        return None
    return results["matches"][0]["metadata"]["text"]

# === Generate answer using Groq ===
def ask_llm(query):
    # Create cache key from query
    cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()

    # Check cache first
    if cache_key in response_cache:
        return response_cache[cache_key]

    context = retrieve_policy(query)
    if not context:
        return "No relevant policy found in the database."

    prompt = (
        f"Answer this prompt as a company IT security resource.\n"
        f"Use the provided policy text below as the basis of your response.\n"
        f"Be concise and factual. The goal is to rephrase the given policy text to more digestible terms or phrasing.\n\n"
        f"Format the response as 3 sentences maximum.\n\n"
        f"--- POLICY TEXT ---\n{context}\n"
        f"-------------------\n\n"
        f"Question: {query}\nAnswer:"
    )

    # Call Groq API
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",  # Very fast and high quality
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful cybersecurity policy assistant. Provide clear, concise answers based only on the provided policy text."
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
        return f"API Error: {str(e)}\n\nPlease check your Groq API key and internet connection."
    except Exception as e:
        return f"Error: {str(e)}"

# === Background Thread for Inference ===
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

# === GUI Application ===
class ChatBotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Chatbot (Groq - Lightning Fast)")
        self.setGeometry(100, 100, 500, 600)
        self.layout = QVBoxLayout()

        defualt_font = QFont("Arial", 14)
        self.setFont(defualt_font)

        # Chat display (read-only)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

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

        self.setLayout(self.layout)

        # Track if we're currently processing
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
        self.chat_display.append(f"<b>Bot:</b> {bot_reply}")

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
