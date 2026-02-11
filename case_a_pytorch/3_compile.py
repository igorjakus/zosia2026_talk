import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import ssl

# Bypass SSL for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "cuda"
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001

print(f"Using device: {DEVICE}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


@torch.compile  # POPRAWKA: Kompilacja modelu (JIT)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(epoch_index):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for _, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    end_time = time.time()
    print(
        f"Epoch {epoch_index} Training Time: {end_time - start_time:.2f}s, Loss: {running_loss / len(trainloader):.4f}"
    )


def validate():
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    with torch.inference_mode():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    print(f"Validation Time: {end_time - start_time:.2f}s, Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    total_start = time.time()
    for epoch in range(EPOCHS):
        train_one_epoch(epoch)
        validate()
    total_end = time.time()
    print(f"Total Execution Time: {total_end - total_start:.2f}s")
