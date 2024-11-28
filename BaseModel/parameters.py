import torch
from model import BERT_LSTM_Model


def load_model_and_parameters(model_path, hidden_dim=128, num_classes=3):
    """
    加载模型权重并查看模型参数
    :param model_path: 模型权重文件路径
    :param hidden_dim: LSTM 隐藏层维度
    :param num_classes: 分类类别数
    """
    model = BERT_LSTM_Model(hidden_dim=hidden_dim, num_classes=num_classes)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("\n--- 模型参数列表 ---")
    for name, param in model.state_dict().items():
        print(f"Parameter: {name}, Shape: {param.shape}")

    print("\n--- 可训练参数列表 ---")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"Trainable Parameter: {name}, Shape: {param.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print("\n--- 参数统计 ---")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")


if __name__ == "__main__":
    model_path = "best_model.pth"
    hidden_dim = 128
    num_classes = 3

    load_model_and_parameters(model_path, hidden_dim, num_classes)