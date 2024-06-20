import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Disable CUDA to avoid GPU-related warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load data
data = pd.read_csv('powerpredict.csv')


def drop_object_columns(df):
    drop_cols = [c for t, c in zip([t != "object" for t in df.dtypes], df.columns) if not t]
    return df.drop(columns=drop_cols)


DOC = drop_object_columns

# Drop non-numeric columns
data = DOC(data)

# Prepare features and target
X = data.drop('power_consumption', axis=1)
y = data['power_consumption']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

# Train and evaluate Random Forest Regressor with default parameters
print("Training Random Forest Regressor with default parameters...")
default_rf = RandomForestRegressor(random_state=42)
default_rf.fit(X_train_sub, y_train_sub)
y_pred_rf_default = default_rf.predict(X_test)
mse_rf_default = mean_squared_error(y_test, y_pred_rf_default)
print(f"Default Random Forest MSE: {mse_rf_default}")

# Hyperparameter tuning for Random Forest Regressor
print("Starting hyperparameter tuning for Random Forest Regressor...")
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_sub, y_train_sub)
best_rf_model = grid_search.best_estimator_
joblib.dump(best_rf_model, 'model_rf.pkl')

# Evaluate the best Random Forest Regressor model
y_pred_rf_best = best_rf_model.predict(X_test)
mse_rf_best = mean_squared_error(y_test, y_pred_rf_best)
print(f"Best Random Forest MSE: {mse_rf_best}")

# Visualize Grid Search results for Random Forest
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
plt.plot(results['param_n_estimators'], -results['mean_test_score'], marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Squared Error')
plt.title('Hyperparameter Tuning for Random Forest')
plt.grid(True)
plt.savefig('rf_hyperparameter_tuning.png')
plt.close()


### _________________________________________________________________________________________________________________________________________________________ ###
### Neural Network regression model

# Define Neural Network model using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=64):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)  # Output a single value for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Set device to CPU
device = torch.device('cpu')

# Prepare data for PyTorch
X_train_tensor = torch.tensor(X_train_sub, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_sub.values, dtype=torch.float32).view(-1, 1).to(device)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Function to train the neural network
def train_nn(model, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)  # MSE Loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")
    return model


input_size = X_train_sub.shape[1]

# Train and evaluate Neural Network with default parameters
print("Training Neural Network with default parameters...")
default_nn_model = NeuralNetwork(input_size, hidden_size1=64, hidden_size2=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(default_nn_model.parameters(), lr=0.001)
trained_default_nn_model = train_nn(default_nn_model, criterion, optimizer, epochs=10)

# Evaluate the default Neural Network model
default_nn_model.eval()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    val_outputs_default = default_nn_model(X_val_tensor)
    mse_nn_default = criterion(val_outputs_default, y_val_tensor).item()
print(f"Default Neural Network MSE: {mse_nn_default}")

# Hyperparameters for tuning
param_dist = {
    'hidden_size1': [32, 64],
    'hidden_size2': [32, 64],
    'learning_rate': [0.001, 0.0025, 0.005, 0.01]
}

results_nn = []

best_nn_model = None
best_loss = float('inf')

# Hyperparameter tuning loop
print("Starting hyperparameter tuning for Neural Network...")
for hidden_size1 in param_dist['hidden_size1']:
    for hidden_size2 in param_dist['hidden_size2']:
        for lr in param_dist['learning_rate']:
            print(f"Training with hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, learning_rate={lr}")
            model = NeuralNetwork(input_size, hidden_size1, hidden_size2).to(device)
            criterion = nn.MSELoss()  # MSE Loss for regression
            optimizer = optim.Adam(model.parameters(), lr=lr)
            trained_model = train_nn(model, criterion, optimizer, epochs=5)  # Reduced epochs for quicker tuning

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()  # Validation loss

                results_nn.append((hidden_size1, hidden_size2, lr, val_loss))

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_nn_model = model
                    print(
                        f"New best model found: hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, learning_rate={lr}, val_loss={val_loss}")

# Save the best neural network model
torch.save(best_nn_model.state_dict(), 'model_nn.pth')

# Convert results to DataFrame for visualization
results_nn_df = pd.DataFrame(results_nn, columns=['hidden_size1', 'hidden_size2', 'learning_rate', 'val_loss'])

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot hidden_size1 vs val_loss
for hidden_size2 in param_dist['hidden_size2']:
    subset = results_nn_df[results_nn_df['hidden_size2'] == hidden_size2]
    ax[0].plot(subset['hidden_size1'], subset['val_loss'], marker='o', label=f'hidden_size2={hidden_size2}')
ax[0].set_xlabel('hidden_size1')
ax[0].set_ylabel('Validation Loss')
ax[0].set_title('Hidden Size 1 vs Validation Loss')
ax[0].legend()
ax[0].grid(True)

# Plot hidden_size2 vs val_loss
for hidden_size1 in param_dist['hidden_size1']:
    subset = results_nn_df[results_nn_df['hidden_size1'] == hidden_size1]
    ax[1].plot(subset['hidden_size2'], subset['val_loss'], marker='o', label=f'hidden_size1={hidden_size1}')
ax[1].set_xlabel('hidden_size2')
ax[1].set_ylabel('Validation Loss')
ax[1].set_title('Hidden Size 2 vs Validation Loss')
ax[1].legend()
ax[1].grid(True)

# Plot learning_rate vs val_loss
for hidden_size1 in param_dist['hidden_size1']:
    for hidden_size2 in param_dist['hidden_size2']:
        subset = results_nn_df[
            (results_nn_df['hidden_size1'] == hidden_size1) & (results_nn_df['hidden_size2'] == hidden_size2)]
        ax[2].plot(subset['learning_rate'], subset['val_loss'], marker='o',
                   label=f'hs1={hidden_size1}, hs2={hidden_size2}')
ax[2].set_xlabel('Learning Rate')
ax[2].set_ylabel('Validation Loss')
ax[2].set_title('Learning Rate vs Validation Loss')
ax[2].set_xscale('log')
ax[2].legend()
ax[2].grid(True)

plt.tight_layout()
plt.savefig('nn_hyperparameter_tuning.png')
plt.close()

# Save results to CSV for reporting
results_nn_df.to_csv('nn_hyperparameter_tuning_results.csv', index=False)

# Evaluate the best Neural Network model
best_nn_model.eval()
with torch.no_grad():
    val_outputs_best = best_nn_model(X_val_tensor)
    mse_nn_best = criterion(val_outputs_best, y_val_tensor).item()
print(f"Best Neural Network MSE: {mse_nn_best}")

# Create a comparison table
comparison_data = {
    'Model': ['Random Forest', 'Random Forest', 'Neural Network', 'Neural Network'],
    'Configuration': ['Default', 'Best', 'Default', 'Best'],
    'MSE': [mse_rf_default, mse_rf_best, mse_nn_default, mse_nn_best]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('model_comparison.csv', index=False)

print(comparison_df)
