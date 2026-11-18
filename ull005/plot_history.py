import json
import matplotlib.pyplot as plt

with open("results/resnet50_history.json", "r") as f:
    history = json.load(f)

epochs = [x["epoch"] for x in history]
train_loss = [x["train_loss"] for x in history]
val_loss = [x["val_loss"] for x in history]
train_f1 = [x["train_f1"] for x in history]
val_f1 = [x["val_f1"] for x in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_loss, 'b-o', label='Train Loss')
ax1.plot(epochs, val_loss, 'r-o', label='Val Loss')
ax1.set_title('ResNet-50 Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, train_f1, 'b-o', label='Train F1')
ax2.plot(epochs, val_f1, 'r-o', label='Val F1')
ax2.set_title('ResNet-50 Macro-F1 per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("results/Figure_4_Training_Curves.png", dpi=300)
plt.show()   