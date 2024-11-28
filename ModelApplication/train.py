import logging
import os

import torch
import torch.nn as nn
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_data, create_data_loaders
from model import BERT_LSTM_Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 创建日志文件夹和文件
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/f1training.log_4",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

# 训练单个周期的函数
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()  # 切换到训练模式
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    # 使用 tqdm 包裹 data_loader 添加进度条
    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失和准确率
        total_loss += loss.item()
        correct_predictions += torch.sum(preds == labels)

        # 收集预测值和标签用于计算 F1 分数
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1


# 验证模型的函数
def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()  # 切换到评估模式
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 使用 tqdm 包裹 data_loader 添加进度条
        for batch in tqdm(data_loader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            # 记录损失和准确率
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels)

            # 收集预测值和标签用于计算 F1 分数
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1


# 主训练函数
# def train(model, train_loader, val_loader, epochs, lr, device):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.CrossEntropyLoss().to(device)
#
#     for epoch in range(epochs):
#         logging.info(f'Epoch {epoch + 1}/{epochs}')
#         print(f'Epoch {epoch + 1}/{epochs}')
#
#         # 训练周期
#         train_loss, train_acc, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, device)
#         logging.info(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
#         print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
#
#         # 验证周期
#         val_loss, val_acc, val_f1 = eval_model(model, val_loader, loss_fn, device)
#         logging.info(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}')
#         logging.info('-' * 30)
#         print(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}')
#         print('-' * 30)
def train(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # 使用AdamW
    loss_fn = nn.CrossEntropyLoss().to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    patience_threshold = 3  # 如果验证损失未改善的周期数阈值

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

    for epoch in range(epochs):
        logging.info(f'Epoch {epoch + 1}/{epochs}')
        print(f'Epoch {epoch + 1}/{epochs}')

        # 训练周期
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, device)
        logging.info(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')

        # 验证周期
        val_loss, val_acc, val_f1 = eval_model(model, val_loader, loss_fn, device)
        logging.info(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}')
        logging.info('-' * 30)
        print(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}')
        print('-' * 30)

        # 更新学习率
        scheduler.step(val_loss)

        # # 检查验证损失以实现早停
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        #     # 保存最佳模型
        #     torch.save(model.state_dict(), 'best_model.pth')
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience_threshold:
        #         print("Early stopping triggered.")
        #         break  # 早停


if __name__ == "__main__":
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    texts, labels = load_data("annotated_news.json")
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 初始化BERT tokenizer和数据加载器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader, val_loader = create_data_loaders(train_texts, train_labels, val_texts, val_labels, tokenizer)

    # 初始化模型
    model = BERT_LSTM_Model(hidden_dim=128, num_classes=3)
    model = model.to(device)

    # 设置训练参数
    epochs = 24
    learning_rate = 2e-5

    # 开始训练
    train(model, train_loader, val_loader, epochs, learning_rate, device)