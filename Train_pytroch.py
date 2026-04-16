print("Importation en cours...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
print("Tout importé !")

# --- Device (utilise le GPU Apple Silicon si disponible) ---
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Device utilisé : {device}")

# --- Données ---
transform = transforms.Compose([
    transforms.ToTensor(),  # convertit en [0,1] automatiquement
])

full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split train / validation (85% / 15%)
val_size = int(0.15 * len(full_train))
train_size = len(full_train) - val_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)

print(f"{train_size} train samples, {val_size} val samples")

# --- Modèle (même architecture que l'original) ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5),   # -> (2, 24, 24)
            nn.ReLU(),
            nn.BatchNorm2d(2),
            nn.MaxPool2d(2),                   # -> (2, 12, 12)

            nn.Conv2d(2, 4, kernel_size=3),   # -> (4, 10, 10)
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2),                   # -> (4, 5, 5)

            nn.Flatten(),                      # -> 100
            nn.Dropout(0.5),
            nn.Linear(100, 10),               # -> 10 classes
        )

    def forward(self, x):
        return self.net(x)

model = CNN().to(device)
print(model)

# --- Entraînement ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
patience = 2
best_val_loss = float("inf")
patience_counter = 0

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, epochs + 1):
    # -- Train --
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y)
        train_correct += (out.argmax(1) == y).sum().item()
        train_total += len(y)

    # -- Validation --
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * len(y)
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += len(y)

    tl = train_loss / train_total
    vl = val_loss / val_total
    ta = train_correct / train_total
    va = val_correct / val_total

    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["train_acc"].append(ta)
    history["val_acc"].append(va)

    print(f"Epoch {epoch:02d} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")

    # -- Early stopping --
    if vl < best_val_loss:
        best_val_loss = vl
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping à l'epoch {epoch}")
            break

# --- Evaluation finale ---
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        test_correct += (out.argmax(1) == y).sum().item()
        test_total += len(y)

print(f"\nTest Accuracy : {test_correct / test_total:.4f}")
torch.save(model.state_dict(), "final_model.pt")

# --- Visualisation ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"],   label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"],   label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()