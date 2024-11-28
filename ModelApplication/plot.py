import re
import matplotlib.pyplot as plt

log_file_path = "logs/f1training.log_4"  # 更新为正确的日志文件路径

epochs = []
train_loss, train_accuracy, train_f1 = [], [], []
val_loss, val_accuracy, val_f1 = [], [], []

# 更新正则表达式以匹配 F1 分数
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
            train_f1.append(float(train_match.group(3)))  # 解析 F1 分数

        val_match = val_pattern.search(line)
        if val_match:
            val_loss.append(float(val_match.group(1)))
            val_accuracy.append(float(val_match.group(2)))
            val_f1.append(float(val_match.group(3)))  # 解析 F1 分数

min_length = min(len(epochs), len(train_loss), len(train_accuracy),
                 len(train_f1), len(val_loss), len(val_accuracy), len(val_f1))

plt.figure(figsize=(15, 5))

# 训练指标图
plt.subplot(1, 2, 1)
plt.plot(epochs[:min_length], train_loss[:min_length], label="Train Loss", color="blue")
plt.plot(epochs[:min_length], train_accuracy[:min_length], label="Train Accuracy", color="green")
plt.plot(epochs[:min_length], train_f1[:min_length], label="Train F1 Score", color="orange")  # 添加 F1 分数曲线
plt.xlabel("Epoch")
plt.ylabel("Train Metrics")
plt.title("Training Metrics over Epochs")
plt.legend()

# 验证指标图
plt.subplot(1, 2, 2)
plt.plot(epochs[:min_length], val_loss[:min_length], label="Validation Loss", color="blue")
plt.plot(epochs[:min_length], val_accuracy[:min_length], label="Validation Accuracy", color="green")
plt.plot(epochs[:min_length], val_f1[:min_length], label="Validation F1 Score", color="orange")  # 添加 F1 分数曲线
plt.xlabel("Epoch")
plt.ylabel("Validation Metrics")
plt.title("Validation Metrics over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
