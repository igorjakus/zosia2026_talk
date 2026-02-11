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

torch.set_float32_matmul_precision("high")

transform = transforms.Compose(
    [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

print("Loading dataset to VRAM...")
train_raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_raw = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


def move_to_vram(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.to(DEVICE)


train_images, train_labels = move_to_vram(train_raw)
test_images, test_labels = move_to_vram(test_raw)
print(
    f"Dataset cached in VRAM. Memory used for data: {train_images.element_size() * train_images.nelement() / 1e9:.2f} GB"
)

model = torchvision.models.resnet50(num_classes=10).to(DEVICE)
model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler("cuda")


def train_one_epoch(epoch, is_warmup=False):
    model.train()
    start_time = time.time()

    indices = torch.randperm(len(train_images), device=DEVICE)

    for i in range(0, len(train_images), BATCH_SIZE):
        batch_idx = indices[i : i + BATCH_SIZE]
        inputs = train_images[batch_idx]
        labels = train_labels[batch_idx]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    label = "Warmup Epoch" if is_warmup else f"Epoch {epoch}"
    print(f"{label} Training Time: {time.time() - start_time:.2f}s")


def validate():
    model.eval()
    start_time = time.time()
    with torch.inference_mode():
        with torch.amp.autocast("cuda"):
            for i in range(0, len(test_images), BATCH_SIZE):
                inputs = test_images[i : i + BATCH_SIZE]
                labels = test_labels[i : i + BATCH_SIZE]
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
