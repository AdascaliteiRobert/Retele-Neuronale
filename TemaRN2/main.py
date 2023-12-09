import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import requests
import pickle
import gzip
import numpy as np
from sklearn.metrics import accuracy_score, f1_score



url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)

# incarcare set de date
with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

# Concatenare set de antrenare si set de testare
train_x, train_y = train_set
test_x, test_y = test_set
train_x = np.concatenate((train_x, test_x), axis=0)
train_y = np.concatenate((train_y, test_y), axis=0)

# Folosirea doar a setului de validare pentru evaluare
valid_x, valid_y = valid_set

# Conversie la tensori
train_x, train_y = torch.Tensor(train_x), torch.LongTensor(train_y)
valid_x, valid_y = torch.Tensor(valid_x), torch.LongTensor(valid_y)

# Definirea modelului MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initializare hiperparametri
input_size = 28 * 28  # Dimensiunea imaginii
hidden_size = 128
output_size = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Creare DataLoader pentru setul de antrenare
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initializare model, functie de cost si optimizator
model = MLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Antrenare model
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # Reshape imaginile la dimensiunea corecta
        batch_x = batch_x.view(-1, input_size)

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass si optimizare
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluare pe setul de antrenare
with torch.no_grad():
    train_outputs = model(train_x.view(-1, input_size))
    _, train_predictions = torch.max(train_outputs, 1)

# Evaluare pe setul de validare
with torch.no_grad():
    valid_outputs = model(valid_x.view(-1, input_size))
    _, valid_predictions = torch.max(valid_outputs, 1)

# Calculare acuratete si F1-score


train_accuracy = accuracy_score(train_y, train_predictions)
valid_accuracy = accuracy_score(valid_y, valid_predictions)

train_f1 = f1_score(train_y, train_predictions, average='weighted')
valid_f1 = f1_score(valid_y, valid_predictions, average='weighted')

print(f'Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
print(f'Validation Accuracy: {valid_accuracy:.4f}, Validation F1: {valid_f1:.4f}')

