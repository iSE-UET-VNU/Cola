import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, n_labels):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(128, 32)
        self.dropout5 = nn.Dropout(0.3)
        self.fc6 = nn.Linear(32, n_labels)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout5(x)
        x = torch.softmax(self.fc6(x), dim=1)
        return x

class Detector:
    def __init__(self, original_data, corrupted_label_index, n_label, dataset_name = None):
        self.original_data = original_data
        self.n_instances = len(original_data)
        self.n_label = n_label
        self.dataset_name = dataset_name
        self.corrupted_label_index = corrupted_label_index
        self.n_corrupted_labels = len(corrupted_label_index)
        self.noise_index = pd.Index([])
        
    def local_detection(self, k_neighbors):

        X = self.original_data.iloc[:, :-1].values
        y = self.original_data.iloc[:, -1].values

        knn_model = KNeighborsClassifier(n_neighbors=k_neighbors+1)
        knn_model.fit(X, y)

        distances, neighbor_indices = knn_model.kneighbors(X)

        filtered_indices = neighbor_indices[:, 1:k_neighbors+1]
        filtered_distances = distances[:, 1:k_neighbors+1]

        predicted_labels = []
        prediction_probabilities = []

        for i in range(len(filtered_indices)):
            neighbor_labels = y[filtered_indices[i]]
            neighbor_distances = filtered_distances[i]

            weight_sum = np.sum(1 / (neighbor_distances ** 2 + 1e-10))
            weighted_votes = np.zeros(self.n_label)

            for j in range(len(neighbor_labels)):
                weighted_votes[neighbor_labels[j]] += 1 / (neighbor_distances[j] ** 2 + 1e-10)

            weighted_prob = weighted_votes / weight_sum
            prediction_probabilities.append(weighted_prob)
            predicted_labels.append(np.argmax(weighted_prob))
            
        predicted_labels = np.array(predicted_labels)
        self.noise_index = self.original_data[predicted_labels != y].index
        self.print_results(phase='local')

    def global_detection(self, n_iterations: int):
        best_model_dir = 'model_saving'
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        if n_iterations == -1:
            n_iterations = 999
            mem = len(self.noise_index)
        for iteration in range(n_iterations):
            best_model_path = os.path.join(best_model_dir, f'best_model_{iteration}.pt')
            best_val_loss = float('inf')
            patience = 10
            early_stopping_counter = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('Device:', device)
            df = self.original_data.drop(self.noise_index)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            X_test = self.original_data.iloc[self.noise_index].iloc[:, :-1]
            y_test = self.original_data.iloc[self.noise_index].iloc[:, -1]

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)
            X_test = torch.tensor(X_test.values, dtype=torch.float32)
            y_test = torch.tensor(y_test.values, dtype=torch.long)

            model = NeuralNetwork(X_train.shape[1], self.n_label).to(device)
            optimizer = optim.AdamW(model.parameters())
            criterion = torch.nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
            epochs = 200
            batch_size = 512
            train_data = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_data, batch_size=batch_size)

            for epoch in range(epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val.to(device))
                    val_loss = criterion(val_outputs, y_val.to(device)).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    try:
                        torch.save(model.state_dict(), best_model_path)
                    except Exception as e:
                        print(f"Cannot save the model: {e}")
                else:
                    early_stopping_counter += 1
                    scheduler.step(val_loss)
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}")

            try:
                model.load_state_dict(torch.load(best_model_path, weights_only=True))
            except:
                print('Failed to load the best model.')

            model.eval()
            with torch.no_grad():
                pred = model(X_test.to(device))
                pred_labels = pred.argmax(dim=1)

            self.noise_index = self.noise_index[(pred_labels.cpu() != y_test.cpu()).numpy()]
            self.print_results(phase='global', iteration=iteration)
            if len(self.noise_index) < mem - 5:
                mem = len(self.noise_index)
            else:
                break
        try:
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
                print(f"Folder {best_model_dir} has been deleted successfully.")
        except Exception as e:
            print(f"Can not remove folder: {e}")
        
    def print_results(self, phase, iteration=0):
        correct_detection_indices = [index for index in self.noise_index if index in self.corrupted_label_index]
        wrong_detection_indices = [index for index in self.noise_index if index not in self.corrupted_label_index]

        precision = len(correct_detection_indices) / len(self.noise_index)
        recall = len(correct_detection_indices) / self.n_corrupted_labels
        F1_score = 2 * precision * recall / (precision + recall)

        print('--------------------------------------------')
        if phase == 'global':
            print('Global Results:')
        else:
            print('Local Results:')
        print(f'Iteration: {iteration + 1}, Dataset: {self.dataset_name}')
        print(f'Precision: {round(precision, 3)}')
        print(f'Recall: {round(recall, 3)}')
        print(f'F1: {round(F1_score, 3)}')
        print(f'# of wrong detected error instances: {len(wrong_detection_indices)}')
        print(f'# of true detected error instances: {len(correct_detection_indices)}')
        print(f'# of current noisy instances: {len(self.noise_index)}')
        print('--------------------------------------------\n\n')