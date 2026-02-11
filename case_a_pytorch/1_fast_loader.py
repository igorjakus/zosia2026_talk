import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = "cuda"
BATCH_SIZE = 512
EPOCHS = 3
LEARNING_RATE = 0.001

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=4, pin_memory=True, persistent_workers=True
)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=4, pin_memory=True, persistent_workers=True
)

model = torchvision.models.resnet50(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(epoch):
    model.train()
    start_time = time.time()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Training Time: {time.time() - start_time:.2f}s")

def validate():
    model.eval()
    start_time = time.time()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        outputs = model(inputs)
    print(f"Validation Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    t0 = time.time()
    for epoch in range(EPOCHS):
        train_one_epoch(epoch)
        validate()
    print(f"Total Execution Time: {time.time() - t0:.2f}s")
