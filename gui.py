import tkinter as tk
from tkinter import scrolledtext

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("IS Security Chatbot")

        self.chat_history = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=50, height=20, font=("Arial", 10))
        self.chat_history.pack(padx=10, pady=10)
        self.chat_history.config(state=tk.DISABLED) # Disable editing

        self.user_input = tk.Entry(master, width=40, font=("Arial", 10))
        self.user_input.pack(padx=10, pady=5, side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message) # Bind Enter key

        self.send_button = tk.Button(master, text="Send", command=self.send_message, font=("Arial", 10))
        self.send_button.pack(padx=5, pady=5, side=tk.RIGHT)

    def send_message(self, event=None):
        user_message = self.user_input.get()
        if user_message:
            self.display_message("You: " + user_message, "user")
            self.user_input.delete(0, tk.END)
            
            # Simulate chatbot response (replace with your actual chatbot logic)
            chatbot_response = self.get_chatbot_response(user_message)
            self.display_message("Chatbot: " + chatbot_response, "chatbot")

    def display_message(self, message, sender):
        self.chat_history.config(state=tk.NORMAL) # Enable editing to insert text
        if sender == "user":
            self.chat_history.insert(tk.END, message + "\n", "user_tag")
        else:
            self.chat_history.insert(tk.END, message + "\n", "chatbot_tag")
        self.chat_history.config(state=tk.DISABLED) # Disable editing again
        self.chat_history.see(tk.END) # Scroll to the end

    def get_chatbot_response(self, user_message):
        # Implement your chatbot logic here
        # This is a placeholder for a simple response
        if "hello" in user_message.lower():
            return "Hi there!"
        elif "how are you" in user_message.lower():
            return "I'm a computer program, so I don't have feelings, but I'm functioning well!"
        else:
            return "I'm not sure how to respond to that."

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
    
    