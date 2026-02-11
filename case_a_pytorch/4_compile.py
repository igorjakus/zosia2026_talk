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

torch.set_float32_matmul_precision('high')

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
model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(epoch, is_warmup=False):
    model.train()
    start_time = time.time()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    label = "Warmup Epoch" if is_warmup else f"Epoch {epoch}"
    print(f"{label} Training Time: {time.time() - start_time:.2f}s")

def validate():
    model.eval()
    start_time = time.time()
    with torch.inference_mode():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
    print(f"Validation Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # WARMUP
    print("Starting Warmup...")
    train_one_epoch(0, is_warmup=True)
    validate()
    
    t0 = time.time()
    # REAL TRAINING
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)
        validate()
    print(f"Total Execution Time: {time.time() - t0:.2f}s")
