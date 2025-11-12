from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QProgressDialog
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# === Pinecone setup ===
pc = Pinecone(api_key="pcsk_6e2JzS_PZYtxdavrAwT6JVaBzFVB2gNs4j6DVeu8LQwNf8WfoT3bwmrbQSCWpHHBvWh91P")
index_name = "infosec-policies"
index = pc.Index(index_name)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === TinyLlama setup ===
print("Loading TinyLlama model... (first load may take a minute)")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype=torch.float16
)

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

# === Generate TinyLlama answer ===
def ask_tinyllama(query):
    context = retrieve_policy(query)
    if not context:
        return "No relevant policy found in the database."

    prompt = (
        f"You are a cybersecurity policy assistant.\n"
        f"Only answer using the text below.\n"
        f"Be concise and factual.\n\n"
        f"--- POLICY TEXT ---\n{context}\n"
        f"-------------------\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response

'''
# === Chat loop ===
print("\n Cybersecurity Policy Chatbot (TinyLlama + Pinecone)")
print("Type your question (or 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    answer = ask_tinyllama(user_input)
    print(f"Bot: {answer}\n")
'''

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
        
        #self.show_processing_dialog()
        
        answer = ask_tinyllama(message)
        
        #self.hide_processing_dialog()
        
        return answer
    
    def show_processing_dialog(self):
        self.progress_dialog = QProgressDialog(self)
        self.progress_dialog.setModal(True)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setWindowTitle("Please Wait")
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.resize(300, 150)
        self.progress_dialog.show()
        
    def hide_processing_dialog(self):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatBotGUI()
    window.show()
    sys.exit(app.exec_())
