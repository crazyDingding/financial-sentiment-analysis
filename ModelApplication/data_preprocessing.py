import chardet
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding, BertTokenizer
import json


class FinancialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        """
        :param texts: List of sentences
        :param labels: List of sentiment labels
        :param tokenizer: BERT tokenizer instance
        :param max_len: Maximum sequence length for BERT
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def detect_encoding(filepath):
    """
    Detect the file encoding.
    :param filepath: Path to the dataset file
    :return: Detected encoding
    """
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def load_data(filepath):
    """
    Load and preprocess the Sentences_AllAgree.txt data.
    :param filepath: Path to the dataset file
    :return: texts and labels
    """
    # 检测文件编码
    encoding = detect_encoding(filepath)
    texts = []
    labels = []
    # 读取 JSON 文件
    with open(filepath, 'r', encoding=encoding) as file:
        data = json.load(file)  # 加载 JSON 数据

        for entry in data:
            text = entry.get("text", "").strip()  # 获取文本
            sentiment = entry.get("sentiment", "").strip().lower()  # 获取情感

            # 将情感标签转换为整数
            if sentiment == 'positive':
                label = 2
            elif sentiment == 'neutral':
                label = 1
            elif sentiment == 'negative':
                label = 0
            else:
                continue  # 如果没有有效标签则跳过

            texts.append(text)  # 添加文本
            labels.append(label)  # 添加标签

    return texts, labels

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    # 对 input_ids 和 attention_mask 进行填充
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 确保所有输入长度相同
    max_len = input_ids.size(1)
    input_ids = torch.stack([torch.cat([seq, torch.zeros(max_len - len(seq))], dim=0) for seq in input_ids])
    attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_len - len(mask))], dim=0) for mask in attention_mask])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
    }

def create_data_loaders(train_texts, train_labels, val_texts, val_labels, tokenizer, batch_size=16, max_len=128):
    """
    Create data loaders for training and validation sets.
    :param train_texts: List of training texts
    :param train_labels: List of training labels
    :param val_texts: List of validation texts
    :param val_labels: List of validation labels
    :param tokenizer: BERT tokenizer
    :param batch_size: Batch size for DataLoader
    :param max_len: Maximum sequence length for BERT
    :return: train_loader and val_loader
    """
    train_dataset = FinancialDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = FinancialDataset(val_texts, val_labels, tokenizer, max_len)

    # 使用 DataCollatorWithPadding 来自动处理填充
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    return train_loader, val_loader


if __name__ == "__main__":
    # 初始化BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 加载数据
    texts, labels = load_data("annotated_news.json")
    # 打印前两行数据以检查提取结果
    print("First two extracted data samples:")
    for i in range(2):
        print(f"Text: {texts[i]}")
        print(f"Label: {labels[i]}")
        print("-" * 50)
    # 将数据分为训练和验证集
    from sklearn.model_selection import train_test_split

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(train_texts, train_labels, val_texts, val_labels, tokenizer)

    print("Data loaders created successfully!")