<<<<<<< HEAD
import os                                                             #provides functions to interact with os (file paths) 
import json                                                           #allows reading of JSON files(intents.json)
import random                                                         

import nltk                                                           #natural language toolkit (tokenization, lemmatization)
import numpy as np                                                    #used to work with arrays

import torch                                                          #PyTorch library for building neural network
import torch.nn as nn                                                 #contains neural network layers (Linear, ReLU)
import torch.nn.functional as F                                       #allows relu to be used directly
import torch.optim as optim                                           #optimization algorithms for training the model
from torch.utils.data import DataLoader, TensorDataset                #utilities to load datasets and create mini-batches

class ChatbotModel(nn.Module):                                         #define the neural architecture for the chatbot

    def __init__(self, input_size, output_size):                       #constructor: initialize the network with input and output dimensions
        super(ChatbotModel, self).__init__()                           #call the constructor of nn.module

        self.fc1 = nn.Linear(input_size, 128)                          #first fully connected layer:    input 128 neurons
        self.fc2 = nn.Linear(128, 64)                                  #second fully connected layer:   input 128 -> 64 neurons 
        self.fc3 = nn.Linear(64, output_size)                          #output layer:                   64 -> number of intents
        self.relu = nn.ReLU()                                          #relu activation function
        self.dropout = nn.Dropout(0.5)                                 #dropout layer to reduce overfitting: 50% of neurons ignored randomly 

    def forward(self, x):                                              #forward pass: defines how input flows through the network
        x = self.relu(self.fc1(x))                                     #apply first linear layer + relu
        x = self.dropout(x)                                            #apply drropout
        x = self.relu(self.fc2(x))                                     #apply second linear layer
        x = self.dropout(x)                                            #apply dropout 
        x = self.fc3(x)                                                #apply output: logits for each intent

        return x                                           


class ChatbotAssistant:                                                #class for managing chatbot logic and interaction

    def __init__(self, intents_path, function_mappings = None):        #constructor: set paths and initialize storage structures
        self.model = None                                              #place holder for neural network model
        self.intents_path = intents_path                               #path to JSON file containing intents
 
        self.documents = []                                            #list of tuples: tokenized patterns and corresponding intent tag
        self.vocabulary = []                                           #list of unique words
        self.intents = []                                              #list of all intent tags
        self.intents_responses = {}                                    #lol

        self.function_mappings = function_mappings                     

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents)) 

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None


def get_stocks():
    stocks = ['APPL', 'META', 'NVDA', 'GS', 'MSFT']

    print(random.sample(stocks, 3))



############################################################### Beginning of Main ###############################################################
if __name__ == '__main__':
    assistant = ChatbotAssistant('intents.json', function_mappings = {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    assistant.save_model('chatbot_model.pth', 'dimensions.json')

    assistant = ChatbotAssistant('intents.json', function_mappings = {'stocks': get_stocks})
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message:')

        if message == '/quit':
            break

        print(assistant.process_message(message))
=======
import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === NLTK Resources ===
# nltk.download('punkt')
# nltk.download('wordnet')

# =====================================================================
# SECTION 1: Neural Network Definition                   
# =====================================================================
class ChatbotModel(nn.Module):
    #Simple feedforward neural network for intent classification
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# =====================================================================
# SECTION 2: Chatbot Assistant Logic
# =====================================================================
class ChatbotAssistant:
    #Handles loading intents, training the model, and processing messages
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.function_mappings = function_mappings or {}

        self.documents = []  # List of (tokenized_pattern, intent_tag)
        self.vocabulary = []  # Unique words across all patterns
        self.intents = []     # List of intent tags
        self.responses = {}   # Maps intent -> list of possible responses
        self.X, self.y = None, None  # Training data

    # -----------------------------------------------------------------
    # Tokenization & Bag-of-Words
    # -----------------------------------------------------------------
    @staticmethod
    def tokenize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(w.lower()) for w in words]

    def bag_of_words(self, words):
        #Convert tokenized words into a binary bag-of-words vector
        return [1 if word in words else 0 for word in self.vocabulary]

    # -----------------------------------------------------------------
    # Load and Parse Intents
    # -----------------------------------------------------------------
    def load_intents(self):
        #Load intents from JSON and prepare vocabulary & documents
        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)

        for intent in intents_data["intents"]:
            tag = intent["tag"]
            if tag not in self.intents:
                self.intents.append(tag)
                self.responses[tag] = intent["responses"]

            for pattern in intent["patterns"]:
                pattern_words = self.tokenize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, tag))

        self.vocabulary = sorted(set(self.vocabulary))

    # -----------------------------------------------------------------
    # Prepare Training Data
    # -----------------------------------------------------------------
    def prepare_data(self):
        #Convert patterns to bag-of-words vectors and intent indices
        X_data, y_data = [], []
        for words, tag in self.documents:
            bow = self.bag_of_words(words)
            X_data.append(bow)
            y_data.append(self.intents.index(tag))

        self.X = np.array(X_data)
        self.y = np.array(y_data)

    # -----------------------------------------------------------------
    # Model Training
    # -----------------------------------------------------------------
    def train(self, lr=0.001, epochs=50, batch_size=8):
        #Train the neural network on the prepared data
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(len(self.vocabulary), len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # -----------------------------------------------------------------
    # Save & Load Model
    # -----------------------------------------------------------------
    def save(self, model_path, meta_path):
        torch.save(self.model.state_dict(), model_path)
        with open(meta_path, 'w') as f:
            json.dump({"vocabulary": self.vocabulary, "intents": self.intents}, f)

    def load(self, model_path, meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.vocabulary, self.intents = meta["vocabulary"], meta["intents"]

        self.model = ChatbotModel(len(self.vocabulary), len(self.intents))
        self.model.load_state_dict(torch.load(model_path))

    # -----------------------------------------------------------------
    # Predict & Process Messages
    # -----------------------------------------------------------------
    def predict_intent(self, text):
        words = self.tokenize(text)
        bow = self.bag_of_words(words)
        bow_tensor = torch.tensor([bow], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            output = self.model(bow_tensor)
            intent_idx = torch.argmax(output, dim=1).item()
            return self.intents[intent_idx]

    def process_message(self, message):
        #Return a response for a given message
        intent = self.predict_intent(message)

        if intent in self.responses:
            return random.choice(self.responses[intent])
        elif intent in self.function_mappings:
            return self.function_mappings[intent]()
        else:
            return "Sorry, I don’t understand that. Can you rephrase?"

# =====================================================================
# SECTION 3: Example Function Mappings
# =====================================================================
def get_policy_reference():
    return "According to IS Policy, Section 4.2: All passwords must be at least 12 characters."

# =====================================================================
# SECTION 4: Main Script
# =====================================================================
if __name__ == "__main__":                                                     #standard python idion: only runs when script is executed directly 
    intents_file = "intents.json"
    model_file = "chatbot_model.pth"
    meta_file = "meta.json"

    # Determine if retraining is needed
    train_model = True                                                          #by default assume model needs to be trained (true)
    if os.path.exists(model_file) and os.path.exists(meta_file):                #if chatbot_model.pth and meta.json exists move to next step. if not retrain
        if os.path.getmtime(intents_file) <= os.path.getmtime(meta_file):       #os.path.getmtime() returns last modified time of the file. if intents.json has not been modified more recently than meta.json -> no new training data
            train_model = False                                                 #only runs if both conditions above are true: model + meta files exist && intents file is older or equal in time to meta file.
                                                                                #overrides existing assumption (true) and tells "no training necessary, load existing model"
   
    # Initialize assistant
    assistant = ChatbotAssistant(intents_file, function_mappings={"password_policy": get_policy_reference})   #allows us to make assistant.exaplmpe calls

    # Train or load model
    if train_model:                                 #if train_model = true, build everything from scratch
        print("Training chatbot...")                #informs user that chatbot is training 
        assistant.load_intents()                    #reads intents.json -> pulls all intnets and responses
        assistant.prepare_data()                    #convert text to bag of words vector
        assistant.train(epochs=50)                  #leans words and intent mapping
        assistant.save(model_file, meta_file)       #writes weights to chatbot_model.pth and vocab to meta.json
        print("Training complete!")                 #inform user training is complete
    else:
        assistant.load(model_file, meta_file)       #load weights and metadata from disk, skips training

    # Start interactive chat
    print("Chatbot ready! Type /quit to exit.")     #inform user that chatbot is ready 
    while True:
        msg = input("You: ")                        #prompt user input
        if msg.lower() == "/quit":                  #if user enters /quit the loop is broken and program ends
            break
        print("Bot:", assistant.process_message(msg))    #prints response chosen from intents.json
>>>>>>> main
