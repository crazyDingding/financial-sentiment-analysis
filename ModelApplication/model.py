import torch
import torch.nn as nn
from transformers import BertModel

class BERT_LSTM_Model(nn.Module):
    def __init__(self, hidden_dim, num_classes, bert_model_name='bert-base-uncased'):
        super(BERT_LSTM_Model, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # 解除冻结BERT的最后几层参数
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)  # 添加Dropout层

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_outputs.last_hidden_state)
        lstm_last_hidden = lstm_out[:, -1, :]
        lstm_last_hidden = self.dropout(lstm_last_hidden)  # 进行Dropout处理
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