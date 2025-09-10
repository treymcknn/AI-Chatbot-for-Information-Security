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

###download nltk resouces once if not already downloaded###
#nltk.download('punkt')
#nltk.download('wordnet')

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

        ### if you see this i committed to main