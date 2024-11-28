import re
import matplotlib.pyplot as plt

log_file_path = "logs/bilstmtraining.log"

epochs = []
train_loss, train_accuracy, train_f1 = [], [], []
val_loss, val_accuracy, val_f1 = [], [], []

epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
train_pattern = re.compile(r"Train loss: ([\d.]+), accuracy: ([\d.]+), F1: ([\d.]+)")
val_pattern = re.compile(r"Validation loss: ([\d.]+), accuracy: ([\d.]+), F1: ([\d.]+)")

with open(log_file_path, "r") as f:
    for line in f:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))

        train_match = train_pattern.search(line)
        if train_match:
            train_loss.append(float(train_match.group(1)))
            train_accuracy.append(float(train_match.group(2)))
            train_f1.append(float(train_match.group(3)))

        val_match = val_pattern.search(line)
        if val_match:
            val_loss.append(float(val_match.group(1)))
            val_accuracy.append(float(val_match.group(2)))
            val_f1.append(float(val_match.group(3)))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label="Loss", color="blue")
plt.plot(epochs, train_accuracy, label="Accuracy", color="green")
plt.plot(epochs, train_f1, label="F1 Score", color="red")
plt.xlabel("Epoch")
plt.ylabel("Train Metrics")
plt.title("Training Metrics over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, val_loss, label="Loss", color="blue")
plt.plot(epochs, val_accuracy, label="Accuracy", color="green")
plt.plot(epochs, val_f1, label="F1 Score", color="red")
plt.xlabel("Epoch")
plt.ylabel("Validation Metrics")
plt.title("Validation Metrics over Epochs")
plt.legend()

plt.tight_layout()
plt.show()