import torch
import torch.nn as nn
import torch.optim as optim


class Perceptron(nn.modules):
    def __init__(self, input_size, hidden_size, output_size):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)
        out = self.softmax(fc2_out)

        return out


def batch_loss_acc(predictions, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(predictions, labels)
    _, predicted_labels = torch.max(predictions, 1)
    correct = (predicted_labels == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return loss, accuracy
