import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# ----------------------------
# Load the TinyLlama model
# ----------------------------
print("Loading TinyLlama model... (this may take a minute on first run)")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"  # Automatically chooses CPU or GPU if available
)

# ----------------------------
# Load company policies
# ----------------------------
with open("company_policies.json", "r", encoding="utf-8") as f:
    policies = json.load(f)

policy_texts = [p["text"] for p in policies]

# ----------------------------
# Create embeddings
# ----------------------------
print("Generating embeddings for policy documents...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast model
policy_embeddings = embedder.encode(policy_texts, convert_to_numpy=True)

# ----------------------------
# Store in FAISS index
# ----------------------------
dimension = policy_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(policy_embeddings)

print(f"Stored {len(policies)} documents in FAISS vector index.")

# ----------------------------
# retrieval helper
# ----------------------------
def retrieve_context(query, k=2):
    """Find top-k similar policy texts for the user's question."""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    contexts = [policy_texts[i] for i in indices[0]]
    return "\n".join(contexts)

# ----------------------------
# Chat function
# ----------------------------
def ask_llama(query, similarity_threshold=0.50):
    """
    Generate a response using retrieved company policies.
    If no relevant policy matches the query, return a fallback message.
    """

    # Retrieve context from FAISS
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 1)  # top 1 match
    best_distance = distances[0][0]
    best_index = indices[0][0]

    # Convert distance (smaller = closer) to similarity
    # Normalize to 0–1 range for readability
    similarity = 1 / (1 + best_distance)

    # If no sufficiently close match, return placeholder
    if similarity < similarity_threshold:
        return "No such record found in company policy."

    # Otherwise, retrieve the best policy text
    context = policy_texts[best_index]

    # Structured, minimal prompt
    prompt = (
        f"You are a company policy assistant.\n"
        f"Only answer using the information below.\n"
        f"Be brief and factual (one sentence max).\n"
        f"Do not restate the question.\n\n"
        f"--- Company Policy ---\n{context}\n"
        f"----------------------\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Tokenize + generate
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and clean
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1]
    response = response.strip()

    return response


    # Decode and clean
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip everything before or after the true answer portion
    if "Answer:" in response:
        response = response.split("Answer:")[-1]
    if "Question:" in response:
        response = response.split("Question:")[-1]

    # Final cleanup
    response = response.strip()
    return response


class ChatBotGUI(QWidget):    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Chatbot")
        self.setGeometry(100, 100, 500, 600)
        self.layout = QVBoxLayout()

        # Chat display (read-only)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.layout.addWidget(self.chat_display)

        # User input
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.setFocus() #set focus to the input box
        self.user_input.returnPressed.connect(self.send_message) #bind enter key to send button
        self.layout.addWidget(self.user_input)

        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.send_button)

        self.setLayout(self.layout)

    def send_message(self):
        #check if input is blank and ask user for a question if so.
        user_text = self.user_input.text().strip() #take away blank spaces before and after input
        
        if not user_text:
            self.chat_display.append("<b>Bot:</b> Please enter a question")
            self.user_input.clear()
            return
        
        # Show user message
        self.chat_display.append(f"<b>You:</b> {user_text}")
        self.user_input.clear()

        # Here, connect with your chatbot logic to get a reply
        bot_reply = self.get_bot_reply(user_text)
        self.chat_display.append(f"<b>Bot:</b> {bot_reply}")

    def get_bot_reply(self, message):
        
        return ask_llama(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatBotGUI()
    window.show()
    sys.exit(app.exec_())