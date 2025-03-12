import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformer import TransformerLayer, Encoder, MultiHeadAttention, FeedForward

# 1. 读取 CSV 文件
data = pd.read_csv('data/ETTh1.csv')
data = data.drop('date', axis=1)
# 归一化
samples = data.values
min_vals = np.min(samples, axis=0)
max_vals = np.max(samples, axis=0)
data_normalized = (samples - min_vals) / (max_vals - min_vals)
samples = data_normalized
# samples = samples[:1000]

# 确定输入步长和预测步长
n_input = 12
n_pre = 1
n_channels = samples.shape[1]

# 2. 将模型按8:1:1的比例划分成训练集，验证集，测试集
train_size = int(0.8 * len(samples))
val_size = int(0.1 * len(samples))

train = samples[:train_size]
val = samples[train_size:train_size + val_size]
test = samples[train_size + val_size:]

# 3. 划分输入和输出
def create_dataset(dataset, n_input, n_pre):
    X, Y = [], []
    for i in range(0, len(dataset) - n_input - n_pre + 1, n_pre):
        X.append(dataset[i:i + n_input])
        Y.append(dataset[i + n_input:i + n_input + n_pre])
    return np.array(X), np.array(Y)

train_X, train_Y = create_dataset(train, n_input, n_pre)
val_X, val_Y = create_dataset(val, n_input, n_pre)
test_X, test_Y = create_dataset(test, n_input, n_pre)

def data_loader(X, Y, batch_size, shuffle=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)
    data = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

train_loader = data_loader(train_X, train_Y, batch_size=8, shuffle=True)
val_loader = data_loader(val_X, val_Y, batch_size=8, shuffle=False)
test_loader = data_loader(test_X, test_Y, batch_size=8, shuffle=False)

# 4. 定义Transformer模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_shape):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_shape[1]  # n_channels
        self.seq_len = input_shape[0]    # n_input
        self.n_pre = n_pre  # 预测步数
        
        # 模型参数
        d_model = 256
        n_heads = 8
        d_ff = 1024
        n_layers = 6
        dropout_prob = 0.1
        
        # 输入映射到模型维度
        self.input_projection = nn.Linear(self.input_dim, d_model)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(self.seq_len, 1, d_model))
        
        # 构建编码器
        self_attn = MultiHeadAttention(n_heads, d_model, dropout_prob)
        feed_forward = FeedForward(d_model, d_ff, dropout_prob)
        encoder_layer = TransformerLayer(
            d_model=d_model,
            self_attn=self_attn,
            src_attn=None,
            feed_forward=feed_forward,
            dropout_prob=dropout_prob
        )
        
        self.encoder = Encoder(encoder_layer, n_layers)
        
        # 输出投影层 - 修改为支持多步预测
        self.output_projection = nn.Linear(d_model, self.input_dim * self.n_pre)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)
        
        # 转换输入维度
        x = x.permute(1, 0, 2)
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.positional_encoding
        
        # 通过编码器
        encoder_output = self.encoder(x, None)  # [seq_len, batch_size, d_model]
        
        # 使用最后一个时间步的输出
        decoder_output = encoder_output[-1]  # [batch_size, d_model]
        
        # 投影到多步输出
        output = self.output_projection(decoder_output)  # [batch_size, input_dim * n_pre]
        
        # 重塑为[batch_size, n_pre, input_dim]
        output = output.view(batch_size, self.n_pre, self.input_dim)
        
        return output

# 5. 初始化模型、损失函数和优化器
input_shape = (n_input, n_channels)
model = TimeSeriesTransformer(input_shape).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# 7. 测试模型
model.eval()
with torch.no_grad():
    predictions = []
    for x, y in test_loader:
        prediction = model(x)
        predictions.append(prediction)

    predictions = torch.cat(predictions, dim=0)

# 8. 可视化结果
predictions = predictions.cpu().numpy()
test_Y = test_Y.reshape(-1, n_channels)
predictions = predictions.reshape(-1, n_channels)
mae = mean_absolute_error(test_Y, predictions)  
print(f"Mean Absolute Error: {mae}")
r2 = r2_score(test_Y, predictions)
print(f"R-squared: {r2}")
mse = mean_squared_error(test_Y, predictions)
print(f"Mean Squared Error: {mse}")
start_length =200
end_length = 500
# 绘制结果
fig, axes = plt.subplots(n_channels, 1, figsize=(10, 6 * n_channels))

for i in range(n_channels):
    ax = axes[i]
    ax.plot(test_Y[start_length:end_length, i], label='True')
    ax.plot(predictions[start_length:end_length, i], label='Predicted')
    ax.set_title(f'Channel {i}')
    ax.legend()

plt.tight_layout()
plt.show()
