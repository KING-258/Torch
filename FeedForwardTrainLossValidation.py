import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
class DL(Dataset):
    def __init__(self, x1, x2, x3, x4, x5, y):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.y = y
    def __len__(self):
        return self.x1.shape[0]
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.x3[idx], self.x4[idx], self.x5[idx], self.y[idx]
class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.l1 = nn.Linear(in_features=5, out_features=3, bias=True)
        self.l2 = nn.Linear(in_features=3, out_features=1, bias=True)
    def forward(self, x1, x2, x3, x4, x5):
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x
device = torch.device('cpu')
x1 = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0], dtype=torch.float32).view(-1, 1)
x2 = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1], dtype=torch.float32).view(-1, 1)
x3 = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0], dtype=torch.float32).view(-1, 1)
x4 = torch.tensor([0, 1, 1, 1, 1, 1, 0, 0], dtype=torch.float32).view(-1, 1)
x5 = torch.tensor([1, 0, 0, 0, 1, 1, 0, 1], dtype=torch.float32).view(-1, 1)
y = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.float32).view(-1, 1)
data = DL(x1, x2, x3, x4, x5, y)
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size])
train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=False)
model = FF().to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for ip1, ip2, ip3, ip4, ip5, op in train_loader:
        ip1, ip2, ip3, ip4, ip5, op = ip1.to(device), ip2.to(device), ip3.to(device), ip4.to(device), ip5.to(device), op.to(device)
        optimizer.zero_grad()        
        out = model.forward(ip1, ip2, ip3, ip4, ip5)
        loss = criterion(out, op)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = (out >= 0.5).float()
        correct_train += (predicted == op).sum().item()
        total_train += op.size(0)
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for ip1, ip2, ip3, ip4, ip5, op in val_loader:
            ip1, ip2, ip3, ip4, ip5, op = ip1.to(device), ip2.to(device), ip3.to(device), ip4.to(device), ip5.to(device), op.to(device)
            out = model.forward(ip1, ip2, ip3, ip4, ip5)
            val_loss = criterion(out, op)
            running_val_loss += val_loss.item()
            predicted = (out >= 0.5).float()
            correct_val += (predicted == op).sum().item()
            total_val += op.size(0)
    val_losses.append(running_val_loss / len(val_loader))
    val_accuracies.append(correct_val / total_val)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
              f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label='Train Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('training_loss_accuracy.png')
plt.tight_layout()
plt.show()