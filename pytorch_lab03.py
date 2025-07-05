import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_loader = DataLoader(datasets.MNIST(root='./data', train=True, transform=transform, download=True), batch_size=32)
test_loader = DataLoader(datasets.MNIST(root='./data', train=False, transform=transform, download=True), batch_size=1000)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)    # Fill correct input and output size
        self.fc2 = nn.Linear(64, 10)    # Fill correct input and output size
    def forward(self, x):
        x = F.relu(self.fc1(x))    # Fill correct layer
        return self.fc2(x)    # Fill correct layer

model = Net()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(5):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
end = time.time()
print(f"PyTorch Training time: {end - start:.2f} seconds")

model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(1)
        correct += (pred == y).sum().item()
print(f"Test accuracy: {correct / len(test_loader.dataset):.4f}")