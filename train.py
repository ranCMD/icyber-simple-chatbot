import json # Used to work with JSON data.
from tkinter import Label # The standard Python interface to the Tk GUI toolkit (Python's de facto standard GUI).

from nltk_utils import tokenize, stem, bag_of_words # Imports said functions from nltk_utils.py
import numpy as np # Numerical Python, is a library consisting of multidimensional array objects and a collection of routines for processing those arrays.

import torch # Library for deep learning developed and maintained by Facebook.
from torch import optim # Package implementing various optimisation algorithms.
import torch.nn as nn # Base class used to develop all neural network models.
from torch.utils.data import Dataset, DataLoader # Dataset and loader that allows use of pre-loaded datasets as well as your own data.

from model import NeuralNet # Imports NeuralNet class from model.py

""" 
train.py: The dataset for the AI.
"""

with open('cyber-intents.json', 'r') as f:
    intents = json.load(f)

    all_words = [] # Creates empty list for words.
    tags = [] # Creates empty list for tags.
    xy = [] # Empty list that will hold both patterns and tags.
    
    for intent in intents['intents']: # Loops over all the intents in intents.json
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']: # Loops over all the different patterns.
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words)) # Sorts all words (only unique ones using set).
    tags = sorted(set(tags))
    
    x_train = [] # Empty list of words for training AI.
    y_train = [] # Empty list of tags for training AI.

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label) # CrossEntropyLoss

    x_train = np.array(x_train)
    y_train = np.array(y_train)

# Can be automatically iterated over and to get batch training.
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train) # Store number of samples as length of x train.
        self.x_data = x_train # Equals x train array.
        self.y_data = y_train # Equals y train array.

    #dataset(idx) # Access dataset with an index.
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] # Return data with given index.
        
    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
#print(input_size, len(all_words)) # Testing
#print(output_size, tags) # Testing
learning_rate = 0.001
num_epochs = 2500 # Consistent accuracy with this number. Adjust accordingly (+/- 500).

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) # Num_workers is for multithreading.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Utilises the GPU, otherwise use CPU.
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss() # This criterion combines LogSoftmax and NLLLoss in one single class.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model.
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device=device,dtype=torch.int64)

        # Forward pass.
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward pass and optimiser step.
        optimizer.zero_grad()
        loss.backward() # To calculate backward propergation.
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss = {loss.item():.4f}')

# Dictionary of various data types.
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth" # Training data file.
torch.save(data, FILE) # Serialises and saves to pickled file.

print(f'Training complete. File saved to {FILE}')