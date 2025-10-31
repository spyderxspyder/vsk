#1 – RNN CSV FILE
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load CSV
df = pd.read_csv(&#39;data.csv&#39;)
values = df[&#39;value&#39;].values.astype(float)

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data)-seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

SEQ_LEN = 5
X, y = create_sequences(values, SEQ_LEN)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):

        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses, val_losses = [], []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred.squeeze(), y_val)
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

# Train vs Val loss
plt.figure(figsize=(8,4))
plt.plot(train_losses, label=&#39;Train Loss&#39;)
plt.plot(val_losses, label=&#39;Validation Loss&#39;)
plt.title(&#39;RNN Train vs Validation Loss&#39;)

plt.legend()
plt.show()

# Predicted vs Actual
with torch.no_grad():
    preds = model(X_val).squeeze().numpy()
plt.figure(figsize=(8,4))
plt.plot(y_val, label=&#39;Actual&#39;)
plt.plot(preds, label=&#39;Predicted&#39;)
plt.title(&#39;RNN Predictions vs Actual&#39;)
plt.legend()
plt.show()

#  Residuals and Histogram
residuals = y_val.numpy() - preds
plt.figure(figsize=(8,4))
plt.plot(residuals, color=&#39;r&#39;)
plt.axhline(0, color=&#39;black&#39;, linestyle=&#39;--&#39;)
plt.title(&#39;Prediction Residuals&#39;)
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True, color=&#39;teal&#39;)
plt.title(&#39;Residual Error Distribution&#39;)
plt.show()

# Correlation heatmap of input sequences
corr = pd.DataFrame(X_val.squeeze().numpy()).corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=False, cmap=&#39;coolwarm&#39;)
plt.title(&#39;Input Sequence Correlation&#39;)
plt.show()

# Confusion matrix (convert regression to discrete bins)
bins = np.linspace(min(values), max(values), 5)
y_true_binned = np.digitize(y_val, bins)
y_pred_binned = np.digitize(preds, bins)
cm = confusion_matrix(y_true_binned, y_pred_binned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=&#39;Purples&#39;)
plt.title(&#39;Confusion Matrix (Discretized Regression)&#39;)
plt.show()

#2 – AUTOENCODER
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load data
df = pd.read_csv(&#39;data.csv&#39;)
X = torch.tensor(df.values, dtype=torch.float32)

class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x): return self.encoder(x)

model = AE(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, X)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Loss curve
plt.plot(losses, label=&#39;Training Loss&#39;)
plt.title(&#39;Autoencoder Loss Curve&#39;)
plt.legend()
plt.show()

# Reconstruction comparison
with torch.no_grad():
    rec = model(X).numpy()
err = X.numpy() - rec

plt.plot(X[:,0], label=&#39;Original&#39;)
plt.plot(rec[:,0], label=&#39;Reconstructed&#39;)
plt.legend(); plt.title(&#39;Original vs Reconstructed (Feature 1)&#39;)
plt.show()

# Error heatmap
sns.heatmap(err, cmap=&#39;coolwarm&#39;, center=0)
plt.title(&#39;Reconstruction Error Heatmap&#39;)
plt.show()

# Error histogram
sns.histplot(err.flatten(), bins=50, kde=True)
plt.title(&#39;Reconstruction Error Distribution&#39;)
plt.show()

# Confusion matrix (error region classification)
bins = np.linspace(err.min(), err.max(), 5)
true_binned = np.digitize(X[:,0], bins)
pred_binned = np.digitize(rec[:,0], bins)
cm = confusion_matrix(true_binned, pred_binned)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=&#39;Reds&#39;)
plt.title(&#39;Confusion Matrix (Binned Reconstruction)&#39;)
plt.show()

#3 – RNN TEXT
import torch, torch.nn as nn, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Load and encode text
text = open(&#39;text.txt&#39;,&#39;r&#39;,encoding=&#39;utf-8&#39;).read()
chars = sorted(list(set(text)))
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}
enc = [c2i[c] for c in text]
SEQ_LEN = 40

X, y = [], []
for i in range(len(enc)-SEQ_LEN):
    X.append(enc[i:i+SEQ_LEN])
    y.append(enc[i+SEQ_LEN])
X, y = torch.tensor(X), torch.tensor(y)

class CharRNN(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, x):
        e = self.embed(x)
        out,_ = self.rnn(e)
        return self.fc(out[:,-1,:])

vocab = len(chars)
model = CharRNN(vocab)
opt = torch.optim.Adam(model.parameters(), lr=0.003)
crit = nn.CrossEntropyLoss()
losses, accs = [], []

for epoch in range(30):
    opt.zero_grad()
    out = model(X)
    loss = crit(out, y)
    losses.append(loss.item())
    preds = torch.argmax(out, dim=1)
    acc = (preds == y).float().mean().item()
    accs.append(acc)
    loss.backward()

    opt.step()
    if (epoch+1)%5==0:
        print(f&#39;Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.2f}&#39;)

# Loss &amp; Accuracy
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].plot(losses); ax[0].set_title(&#39;Loss&#39;)
ax[1].plot(accs); ax[1].set_title(&#39;Accuracy&#39;)
plt.show()

# Confusion matrix (chars)
cm = confusion_matrix(y.numpy(), preds.numpy(), labels=list(range(vocab)))
ConfusionMatrixDisplay(cm, display_labels=chars).plot(cmap=&#39;Blues&#39;)
plt.title(&#39;Character Confusion Matrix&#39;)
plt.xticks(rotation=90)
plt.show()

# Probability heatmap
probs = torch.softmax(out, dim=1).detach().numpy()
sns.heatmap(probs[:50], cmap=&#39;YlGnBu&#39;)
plt.title(&#39;Character Probability Heatmap (First 50 Samples)&#39;)
plt.xlabel(&#39;Character Index&#39;)
plt.ylabel(&#39;Sample Index&#39;)
plt.show()
