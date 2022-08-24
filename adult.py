# %%
import numpy as np
import pandas as pd
import torch
# %%
df = pd.read_csv('adult.data')
# %%
df.head()
# %%
def unique_ints(column_name:  str):
    series = df[column_name].map(
        {
            v: k for k, v in enumerate(np.unique(df[column_name]))
        }
    )
    return series
# %%
for column in df.columns:
    if np.dtype(df[column]) == np.dtype('O'):
        df[column] = unique_ints(column_name=column)
# %%
df.describe().transpose()
# %%
X,y =  df.iloc[:, :14].values, df.iloc[:, 14].values
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size=0.8, stratify=y)
# %%
std = StandardScaler()
std.fit(X_train)
X_train_norm = std.transform(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
X_test_norm = std.transform(X_test)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

# %%
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# %%

train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 10
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
# %%
num_epochs = 200
log_epochs = 20
torch.manual_seed(1)
# %%
class Model(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Tanh()(x)
        x = self.layer2(x)
        x = nn.ReLU()(x)
        x = self.layer3(x)
        x = nn.Softmax(dim=1)(x)
        return x
# %%
input_size = X_train_norm.shape[1]
hidden1_size = 20
hidden2_size = 10
output_size = 1
model = Model(
    input_size=input_size, 
    hidden1_size=hidden1_size, 
    hidden2_size=hidden1_size,
    output_size=output_size
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# %%
for epoch in range(num_epochs):
    loss_hist_train = 0
    for X_batch, y_batch in train_dl:
        pred = model(X_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()
    if epoch % log_epochs == 0:
        print(
            f'Epoch {epoch} Loss '
            f'{loss_hist_train/len(train_dl):.4f}'
        )

# %%
with torch.no_grad():
    pred = model(X_test_norm)[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(torch.argmax(pred), y_test).item():.4f}')
# %%
print(f'miss_class: {(y_test != torch.argmax(pred)).sum()}')
# %%
