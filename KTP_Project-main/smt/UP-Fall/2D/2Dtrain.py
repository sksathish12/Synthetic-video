import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from dataset_prep_2d import OpticalFlow2DDataset
import numpy as np
import seaborn as sns
import os

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super(FallDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size = 3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size = 3)
        self.conv3 = nn.Conv2d(128, 64, kernel_size = 3)
        
        self.pool = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(64 * 3 * 4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 254)
        self.fc4 = nn.Linear(254, 2) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
def compute_metrics(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, specificity, f1

def plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses):
    epochs_range = range(1, len(accuracies) + 1)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, accuracies, 'o-', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, precisions, 'o-', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, recalls, 'o-', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, specificities, 'o-', label='Specificity')
    plt.title('Specificity')
    plt.xlabel('Epochs')
    plt.ylabel('Specificity')
    plt.ylim([0, 1])

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, f1_scores, 'o-', label='F1-Score')
    plt.title('F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.ylim([0, 1])
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, val_losses, 'o-', label='Validation Loss', color='red')
    plt.plot(epochs_range, train_losses, 'o-', label='Training Loss', color='blue')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize = (10, 7))
    sns.heatmap(cm, annot = True, fmt = '', cmap='Blues', xticklabels = classes, yticklabels = classes, annot_kws={"size": 18})
    plt.xlabel('Predicted Labels', fontsize = 16)
    plt.ylabel('True Labels', fontsize = 16)
    plt.title('Confusion Matrix',  fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.savefig(f'confusion_matrix.jpg', format='jpg', dpi = 400)
    plt.close()
    
def train_model(dataloader_train, dataloader_val, num_epochs=50, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallDetectionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    best_f1_score = 0
    best_accuracy = 0
    best_epoch = 0
    
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    val_losses = []
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        train_true_labels = []
        train_predictions = []
        
        for batch_features, batch_labels in dataloader_train:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_true_labels.extend(batch_labels.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())
            
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        train_cm = confusion_matrix(train_true_labels, train_predictions)
        print(f"Training Confusion for Epoch {epoch+1}:\n{train_cm}")
        
        model.eval()
        epoch_val_losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader_val:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                epoch_val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        
        if f1 > best_f1_score or (f1 == best_f1_score and accuracy > best_accuracy):
            best_f1_score = f1
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_model_epoch_test.pth')
            print(f"New best model saved at epoch {epoch+1} with F1 Score: {best_f1_score:.4f} and Accuracy: {best_accuracy:.4f}")
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}")
        
    plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses)
        
    return model, train_losses, val_losses, accuracies, precisions, recalls, specificities, f1_scores

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, precision, recall, specificity, f1, all_labels, all_preds

def cross_validate_model(dataset, n_splits = 5, num_epochs = 50, learning_rate = 0.00001):
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_specificities = []
    all_f1_scores = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('----------------------------------')
        
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)
                
        dataloader_train = DataLoader(train_subset, batch_size = 32, shuffle = True)    
        dataloader_val = DataLoader(val_subset, batch_size = 32, shuffle = False)
        
        model, train_losses, val_losses, accuracies, precisions, recalls, specificities, f1_scores = train_model(dataloader_train, dataloader_val, num_epochs, learning_rate)
        
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_accuracies.extend(accuracies)
        all_precisions.extend(precisions)
        all_recalls.extend(recalls)
        all_specificities.extend(specificities)
        all_f1_scores.extend(f1_scores)
    
    mean_accuracy = np.mean(all_accuracies)
    mean_precision = np.mean(all_precisions)    
    mean_recall = np.mean(all_recalls)
    mean_specificity = np.mean(all_specificities)
    mean_f1_score = np.mean(all_f1_scores)
    mean_training_loss = np.mean(all_train_losses)
    mean_val_loss = np.mean(all_val_losses)
    
    std_accuracy = np.std(all_accuracies)
    std_precision = np.std(all_precisions)
    std_recall = np.std(all_recalls)
    std_specificity = np.std(all_specificities)
    std_f1_score = np.std(all_f1_scores)
    std_training_loss = np.std(all_train_losses)
    std_val_loss = np.std(all_val_losses)
    
    print("Cross-validation results:")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}")
    print(f"F1-Score: {mean_f1_score:.4f} ± {std_f1_score:.4f}")
    print(f"Training Loss: {mean_training_loss:.4f} ± {std_training_loss:.4f}")
    print(f"Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    
if __name__ == "__main__":
    features_path = 'D:/nparray_2D_balanced'
    test_path = 'D:/nparray_2D'

    train_val_dataset = OpticalFlow2DDataset(features_path)
    test_dataset = OpticalFlow2DDataset(test_path)

    dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FallDetectionCNN().to(device)
    
    saved_model_path = "3fall_detection_model_2dx.pth"
    
    if os.path.isfile(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
        print("Loaded saved model.")
    
    else:
        print("No saved model found. Training a new model.")
        cross_validate_model(train_val_dataset)
        
    model.eval()
    
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions = evaluate_model(model, dataloader_test, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
    
    plot_confusion_matrix(true_labels, predictions, classes = ["No Fall", "Fall"])
    
    torch.save(model.state_dict(), '3fall_detection_model_2d.pth')

