import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel, QScrollArea
from testlogic import ourlogic

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
        # Placeholder logic — replace with your chatbot's response generation
        return "This is a response to: " + message

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatBotGUI()
    window.show()
    sys.exit(app.exec_())