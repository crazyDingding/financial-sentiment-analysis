import torch
import torch.nn as nn
from transformers import BertModel


class BERT_LSTM_Model(nn.Module):
    def __init__(self, hidden_dim, num_classes, bert_model_name='bert-base-uncased'):
        """
        :param hidden_dim: Hidden dimension size for LSTM
        :param num_classes: Number of output classes
        :param bert_model_name: Name of the pre-trained BERT model
        """
        super(BERT_LSTM_Model, self).__init__()

        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(bert_model_name)

        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name:  # 仅解冻最后一层
                param.requires_grad = True
            else:
                param.requires_grad = False

        # LSTM 层
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True)

        # 分类层
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        # 使用 BERT 获取嵌入
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # BERT 的 last_hidden_state 是形状为 (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(bert_outputs.last_hidden_state)

        # 获取 LSTM 的最后一个时间步的输出
        lstm_last_hidden = lstm_out[:, -1, :]

        # 分类层输出
        output = self.classifier(lstm_last_hidden)

        return output


# 测试模型定义（仅在该文件作为主程序运行时执行）
if __name__ == "__main__":
    model = BERT_LSTM_Model(hidden_dim=128, num_classes=3)
    print(model)

    # 测试输入张量
    input_ids = torch.randint(0, 1000, (2, 128))  # 模拟 batch_size=2, sequence_length=128 的输入
    attention_mask = torch.ones((2, 128), dtype=torch.long)

    # 模型前向传播
    outputs = model(input_ids, attention_mask)
    print("Model output shape:", outputs.shape)  # 输出形状应为 (batch_size, num_classes)