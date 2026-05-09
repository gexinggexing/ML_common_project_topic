import torch
import torch.nn as nn

class ExerciseEEGSimpleRNN(nn.Module):
    """
    完成 SimpleRNN 的 6 个填空:
    1) RNN 输入维度定义（input_size）
    2) RNN 非线性选择（用于减轻梯度消失）
    3) 循环权重初始化（用于稳定梯度）
    4) 输入张量维度变换
    5) 双向 RNN 特征拼接
    6) 梯度裁剪（用于处理梯度爆炸）
    """

    def __init__(
        self,
        chans=20,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True,
        grad_clip=1.0,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.grad_clip = grad_clip

        self.rnn = nn.RNN(
            input_size=____,  # TODO-RNN-1: 改成输入特征维度 chans
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=____,  # TODO-RNN-2: 填 "relu"（减轻梯度消失）
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self._init_rnn_weights()

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _init_rnn_weights(self):
        # 梯度消失/爆炸处理（初始化侧）
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                ____  # TODO-RNN-3: 对循环权重做正交初始化
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x 原始 shape: (B, C, T)
        x = ____  # TODO-RNN-4: 变换为 (B, T, C)

        # h_n: (num_layers * num_directions, B, hidden_dim)
        out, h_n = self.rnn(x)

        if self.bidirectional:
            feat = ____  # TODO-RNN-5: 拼接最后一层双向 hidden state
        else:
            feat = h_n[-1]

        logits = self.classifier(feat)
        return logits

    def clip_gradients(self):
        # 梯度爆炸处理（训练侧）：backward 后执行
        return ____  # TODO-RNN-6: 使用 clip_grad_norm_ 裁剪 self.parameters()


class ExerciseEEGLSTM(nn.Module):
    """
    完成 6 个填空:
    1) LSTM 输入维度定义（input_size）
    2) 输入张量维度变换
    3) LSTM 返回值的正确接收
    4) 双向 LSTM 特征拼接
    5) 单向 LSTM 最后一层 hidden state 取法
    6) 梯度裁剪（用于处理梯度爆炸）
    """

    def __init__(
        self,
        chans=20,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        bidirectional=True,
        grad_clip=1.0,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.grad_clip = grad_clip

        self.lstm = nn.LSTM(
            input_size=____,  # TODO-1: 改成输入特征维度 chans
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x 原始 shape: (B, C, T)
        x = ____  # TODO-2: 变换为 (B, T, C)

        # TODO-3: 正确接收 LSTM 返回的 hidden states
        out, ____ = self.lstm(x)

        if self.bidirectional:
            feat = ____  # TODO-4: 拼接最后一层双向 hidden state
        else:
            feat = ____  # TODO-5: 取最后一层单向 hidden state

        logits = self.classifier(feat)
        return logits

    def clip_gradients(self):
        # 梯度爆炸处理（训练侧）：backward 后执行
        return ____  # TODO-6: 使用 clip_grad_norm_ 裁剪 self.parameters()
