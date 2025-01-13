import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

class MisclassifiedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Ensure that each item in samples is a tuple of (optical_flow_tensor, true_label)
        if len(self.samples[idx]) == 2:
            optical_flow_tensor, true_label = self.samples[idx]
            optical_flow_tensor = torch.tensor(optical_flow_tensor, dtype=torch.float32)
            label_tensor = torch.tensor(true_label, dtype=torch.long)
            return optical_flow_tensor, label_tensor
        else:
            raise ValueError(f"Expected 2 values but got {len(self.samples[idx])}")

    
def get_misclassified_non_falls_and_fall_samples(model, dataloader, device):
    model.eval()
    misclassified_non_falls = []
    misclassified_falls = []
    non_fall_indices = []
    fall_indices = []

    with torch.no_grad():
        for i, (batch_features, batch_labels, _) in enumerate(dataloader):
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            
            # Identify misclassified samples
            misclassified_indices_batch = (predicted != batch_labels).nonzero(as_tuple=True)[0]
            for idx in misclassified_indices_batch:
                global_idx = i * dataloader.batch_size + idx.item()
                if batch_labels[idx].item() == 1:  # Misclassified fall
                    misclassified_falls.append(global_idx)
                else:  # Misclassified non-fall
                    non_fall_indices.append(global_idx)
                    optical_flow_tensor = batch_features[idx].cpu().numpy()
                    true_label = batch_labels[idx].item()
                    misclassified_non_falls.append((optical_flow_tensor, true_label))
                
            # Identify all fall samples
            fall_indices_batch = (batch_labels == 1).nonzero(as_tuple=True)[0]
            for idx in fall_indices_batch:
                global_idx = i * dataloader.batch_size + idx.item()
                fall_indices.append(global_idx)
    
    # Exclude the misclassified fall indices from the list of fall indices
    remaining_fall_indices = list(set(fall_indices) - set(misclassified_falls))
    
    # Split remaining fall samples into 80% for retraining and 20% for testing
    train_fall_indices, test_fall_indices = train_test_split(remaining_fall_indices, test_size=0.2, random_state=42)
    
    # Now, get the actual samples using the correct indices
    train_fall_samples = [dataloader.dataset[i] for i in train_fall_indices]
    test_fall_samples = [dataloader.dataset[i] for i in test_fall_indices]
    
    return misclassified_non_falls, non_fall_indices, train_fall_samples, train_fall_indices, test_fall_samples, test_fall_indices

def retrain_model_on_misclassified_samples(model, misclassified_dataloader, fall_dataloader, num_epochs=10, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_true_labels = []
        epoch_predictions = []
        
        # Train on misclassified non-falls
        for batch_features, batch_labels in misclassified_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            epoch_true_labels.extend(batch_labels.cpu().numpy())
            epoch_predictions.extend(predicted.cpu().numpy())

        # Train on 80% of falls
        for batch_features, batch_labels in fall_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            epoch_true_labels.extend(batch_labels.cpu().numpy())
            epoch_predictions.extend(predicted.cpu().numpy())
            
        accuracy = accuracy_score(epoch_true_labels, epoch_predictions)
        precision = precision_score(epoch_true_labels, epoch_predictions)
        recall = recall_score(epoch_true_labels, epoch_predictions)
        f1 = f1_score(epoch_true_labels, epoch_predictions)    
        
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1-Score: {f1:.4f}")
             
    return model

