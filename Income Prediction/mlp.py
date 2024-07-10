import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

exec(open("data_import.py").read())

# Split data into features and labels
X = adults.drop("value", axis = 1).values
y = adults["value"].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Data conversion to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Defining our ANN model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 32)
        self.fc2 = nn.Linear(64, 128)
        self.fc3_1 = nn.Linear(128, 32)
        self.fc3_2 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x2 = self.sigmoid(self.fc1(x))
        x = torch.cat((x1, x2), dim = 1)
        x = self.relu(self.fc2(x))
        x1 = self.relu(self.fc3_1(x))
        x2 = self.sigmoid(self.fc3_2(x))
        x = torch.cat((x1, x2), dim = 1)
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x
    
# Hyperparameters
model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience = 10)

epochs = 1000
patience = 50
best_loss = float('inf')
counter = 0

# Model Training
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        break
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Model evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    accuracy = (y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])).item()
    print(f"Accuracy: {accuracy:.4f}")