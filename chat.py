import random # To generate random numbers.
import json # Used to work with JSON data.
import torch # Library for deep learning developed and maintained by Facebook.
from model import NeuralNet # Imports NeuralNet class from model.py
from nltk_utils import bag_of_words, tokenize # Imports said functions from nltk_utils.py
import wikipedia as wk # Access and parse data from Wikipedia.
import re, string, unicodedata # Provides regular expression matching operations.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Utilises the GPU, otherwise use CPU.

# Opens file in read mode
with open('cyber-intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "iCyber"
print("iCyber: Hi, I'm a bot who can give you the latest updates on cyber security threats and intel! \niCyber: Ask me a question in plain english. Type 'quit' to exit. \niCyber: Type 'tell me about' (case sensitive) followed by a subject which I will look up for you eg. log4j, malware, virus")
# What the user types in the chat
while True:
    senetence = input('You: ')

    # Ending conversation
    if senetence == "quit":
        break

    # Searching Wikipedia for subject
    if "tell me about" in senetence:
        print("Checking Wikipedia...")
        reg_ex = re.search('tell me about (.*)', senetence)
        try:
            if reg_ex:
                topic = reg_ex.group(1)
                wiki = wk.summary(topic, sentences = 3)
                print(wiki)
        except Exception as e:
            print("No content has been found")

    # General conversation with bot
    elif "tell me about" not in senetence:
        senetence = tokenize(senetence)
        x = bag_of_words(senetence, all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: Sorry, I don't understand that.")

    
