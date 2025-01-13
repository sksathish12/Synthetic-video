import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from dataset_prep_3d_test import OpticalFlow3DDataset
from datetime import datetime
import cv2
import seaborn as sns
import os
import random
import gc
import pandas as pd
from visualize_misclassified import visualize_misclassified_samples, get_misclassified_samples_for_vis, visualize_optical_flow
from misclassified_retrain import retrain_model_on_misclassified_samples, MisclassifiedDataset, get_misclassified_non_falls_and_fall_samples

# Set seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()
gc.collect()

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super(FallDetectionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv3d(2, 64, (3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, (3, 3, 3), padding=1)
        self.bn4 = nn.BatchNorm3d(256)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2) 

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x, 2)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x, 2)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.max_pool3d(x, 2)
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
        
def compute_metrics(true_labels, predictions):
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, specificity, f1

def plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses, fold, input, experiment_dir):
    epochs_range = range(1, len(accuracies) + 1)
    plt.figure(figsize=(15, 10))
    
    # plt.subplot(2, 3, 1)
    # plt.plot(epochs_range, accuracies, 'o-', label='Accuracy')
    # plt.title('Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.ylim([0, 1])

    # plt.subplot(2, 3, 2)
    # plt.plot(epochs_range, precisions, 'o-', label='Precision')
    # plt.title('Precision')
    # plt.xlabel('Epochs')
    # plt.ylabel('Precision')
    # plt.ylim([0, 1])

    # plt.subplot(2, 3, 3)
    # plt.plot(epochs_range, recalls, 'o-', label='Recall')
    # plt.title('Recall')
    # plt.xlabel('Epochs')
    # plt.ylabel('Recall')
    # plt.ylim([0, 1])

    # plt.subplot(2, 3, 4)
    # plt.plot(epochs_range, specificities, 'o-', label='Specificity')
    # plt.title('Specificity')
    # plt.xlabel('Epochs')
    # plt.ylabel('Specificity')
    # plt.ylim([0, 1])

    # plt.subplot(2, 3, 5)
    # plt.plot(epochs_range, f1_scores, 'o-', label='F1-Score')
    # plt.title('F1-Score')
    # plt.xlabel('Epochs')
    # plt.ylabel('F1-Score')
    # plt.ylim([0, 1])
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, val_losses, 'o-', label='Validation Loss')
    plt.plot(epochs_range, train_losses, 'o--', label='Training Loss')
    plt.title('Learning rate 0.0001')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/Fold{fold}_{input}xdd.svg', format='svg', dpi = 400)
    plt.close()


def plot_confusion_matrix(true_labels, predictions, input, experiment_dir, classes):
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize = (10, 7))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap='Blues', xticklabels = classes, yticklabels = classes, annot_kws={"size": 18})
    plt.xlabel('Predicted Labels', fontsize = 16)
    plt.ylabel('True Labels', fontsize = 16)
    plt.title('Confusion Matrix',  fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.savefig(f'{experiment_dir}/Matrix_Fold_{input}.jpg', format='jpg', dpi = 400)
    plt.close()
    
def train_model(dataloader_train, dataloader_val, fold, input, log_path, num_epochs, learning_rate, weight_decay, experiment_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallDetectionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
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
    
    min_epoch_to_save = int(num_epochs * 0.2)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        train_true_labels = []
        train_predictions = []
        
        for batch_features, batch_labels, _ in dataloader_train:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_true_labels.extend(batch_labels.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())
            
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        train_cm = confusion_matrix(train_true_labels, train_predictions)
        #print(f"Training Confusion for Epoch {epoch+1}:\n{train_cm}")
        logging_output(f"Training Confusion for Epoch {epoch+1}:\n{train_cm}", log_path)
       
        model.eval()
        val_true_labels = []
        val_predictions = []
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch_features, batch_labels, _ in dataloader_val:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                epoch_val_losses.append(loss.item())
                
                _, predicted = torch.max(outputs.data, 1)
                val_true_labels.extend(batch_labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        accuracy, precision, recall, specificity, f1 = compute_metrics(val_true_labels, val_predictions)
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        
        if epoch >= min_epoch_to_save:
            if f1 > best_f1_score or (f1 == best_f1_score and accuracy > best_accuracy):
                best_f1_score = f1
                best_accuracy = accuracy
                best_epoch = epoch
                best_model = model.state_dict()
                logging_output(f"New best model at epoch {epoch+1} with F1 Score: {best_f1_score:.4f} and Accuracy: {best_accuracy:.4f}", log_path)
        
        logging_output(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1-Score: {f1:.4f}", log_path)
            
    plot_metrics(accuracies, precisions, recalls, specificities, f1_scores, val_losses, train_losses, fold, input, experiment_dir)
        
    return best_model, train_losses, val_losses, accuracies, precisions, recalls, specificities, f1_scores

def evaluate_model(model, dataloader, criterion, device, file_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_file_names = []
    total_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels, batch_file_names in dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_file_names.extend(batch_file_names)
    
    accuracy, precision, recall, specificity, f1 = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, precision, recall, specificity, f1, all_labels, all_preds, all_file_names

def save_predictions_to_csv(file_names, predictions, labels, output_file):
    df = pd.DataFrame({
        'File Name': file_names,
        'Prediction': predictions
    })
    df.to_csv(output_file, index=False)
    
def cross_validate_model(dataset, input, log_path, model_dir, experiment_dir, n_splits = 5, num_epochs = 50, learning_rate = 0.0001, weight_decay = 1e-5):
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_specificities = []
    all_f1_scores = []
    
    best_overall_f1_score = 0
    best_overall_accuracy = 0
    best_overall_fold = 0
    best_overall_model_state = None
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        logging_output(f'FOLD {fold}', log_path)
        logging_output('----------------------------------', log_path)
        
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)
                
        dataloader_train = DataLoader(train_subset, batch_size = 16, shuffle = True)    
        dataloader_val = DataLoader(val_subset, batch_size = 16, shuffle = False)
        
        best_model, train_losses, val_losses, accuracies, precisions, recalls, specificities, f1_scores = train_model(dataloader_train, dataloader_val, fold, input, log_path, num_epochs, learning_rate, weight_decay, experiment_dir)
        
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_accuracies.extend(accuracies)
        all_precisions.extend(precisions)
        all_recalls.extend(recalls)
        all_specificities.extend(specificities)
        all_f1_scores.extend(f1_scores)

        if max(f1_scores) > best_overall_f1_score or (max(f1_scores) == best_overall_f1_score and max(accuracies) > best_overall_accuracy):
            best_overall_f1_score = max(f1_scores)
            best_overall_accuracy = max(accuracies)
            best_overall_fold = fold
            best_overall_model_state = best_model
        
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
    
    # print("Cross-validation results:")
    # print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    # print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    # print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    # print(f"Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}")
    # print(f"F1-Score: {mean_f1_score:.4f} ± {std_f1_score:.4f}")
    # print(f"Training Loss: {mean_training_loss:.4f} ± {std_training_loss:.4f}")
    # print(f"Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    
    logging_output("Cross-validation results:", log_path)
    logging_output(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}", log_path)
    logging_output(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}", log_path)
    logging_output(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}", log_path)
    logging_output(f"Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}", log_path)
    logging_output(f"F1-Score: {mean_f1_score:.4f} ± {std_f1_score:.4f}", log_path)
    logging_output(f"Training Loss: {mean_training_loss:.4f} ± {std_training_loss:.4f}", log_path)
    logging_output(f"Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}", log_path)
    
    if best_overall_model_state is not None:
        best_model_path = os.path.join(model_dir, f'best_model_overall_{input}.pth')
        torch.save(best_overall_model_state, best_model_path)
        logging_output(f"Best overall model saved from fold {best_overall_fold} with F1 Score: {best_overall_f1_score:.4f} and Accuracy: {best_overall_accuracy:.4f}", log_path)
    
    
          
def logging_output(message, file_path='./def_log.txt'):
    try:
       
    # Write to file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(message)
        with open(file_path, 'a') as file:
            file.write(f'{current_time} : {message} \n')
    except Exception as e:
        print(f"Error logging to {file_path} : {e}")    

if __name__ == "__main__": 
    features_path = 'D:\\downsampled'
    test_path = 'D:\\testtt'
    raw_frames_path = 'D:\\MSC_Project\\UP-FALL-DATASET'
    
    experiment_name = input("Enter experiment name: ")
    experiment_dir = f'graphs/3DKFold/{experiment_name}'
    os.makedirs(experiment_dir, exist_ok=True)
    model_dir = os.path.join(experiment_dir, 'Models')
    os.makedirs(model_dir, exist_ok=True)
    
    log_path = os.path.join(experiment_dir, f'Training_Log_{experiment_name}.log')
    
    logging_output(experiment_name, log_path)
    train_val_dataset = OpticalFlow3DDataset(features_path, raw_frames_path, augment=False)
    test_dataset = OpticalFlow3DDataset(test_path, raw_frames_path, augment=False)
    
    #logging_output(f"Number of aug falls: {train_val_dataset.num_augmented_falls}", log_path)
    
    dataloader_test = DataLoader(test_dataset, batch_size = 16,  shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FallDetectionCNN().to(device)
    
    saved_model_path = os.path.join(model_dir, f'best_model_overall_{experiment_name}.pth')
    
    retrain_flag = False
    visualize_flag = False
    
    if os.path.isfile(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
        logging_output("Loaded saved model.", log_path)
        
        model.eval()
    
        criterion = nn.CrossEntropyLoss().to(device)
        test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions, file_names = evaluate_model(model, dataloader_test, criterion, device, test_dataset.file_names)
        
        # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}")
        logging_output(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}", log_path)
        
        plot_confusion_matrix(true_labels, predictions, experiment_name, experiment_dir, classes = ["No Fall", "Fall"])
        
        output_file = os.path.join(experiment_dir, f'predictions_{experiment_name}.csv')
        save_predictions_to_csv(file_names, predictions, true_labels, output_file)
        logging_output(f"Predictions saved to {output_file}", log_path)
        
        visualize_flag = input("Would you like to visualize misclassified samples? (y/n)").strip().lower() == 'y'
        retrain_flag = input("Would you like to retrain on misclassified samples? (y/n)").strip().lower() == 'y'

        if visualize_flag:
            misclassified_samples_vis, misclassified_indices = get_misclassified_samples_for_vis(model, dataloader_test, device)
            visualize_misclassified_samples(misclassified_samples_vis)
            
        if retrain_flag:
            misclassified_non_falls, non_fall_indices, train_fall_samples, train_fall_indices, test_fall_samples, test_fall_indices = get_misclassified_non_falls_and_fall_samples(model, dataloader_test, device)

            # Prepare the datasets
            misclassified_dataset = MisclassifiedDataset(misclassified_non_falls)
            misclassified_dataloader = DataLoader(misclassified_dataset, batch_size=16, shuffle=True)
            
            train_fall_dataset = MisclassifiedDataset(train_fall_samples)
            fall_dataloader = DataLoader(train_fall_dataset, batch_size=16, shuffle=True)
            
            #freeze_layers = ['conv1', 'conv2', 'bn1', 'bn2']  # Example layers to freeze
            model = retrain_model_on_misclassified_samples(model, misclassified_dataloader, fall_dataloader, num_epochs=10)

            remaining_indices = list(set(range(len(test_dataset))) - set(non_fall_indices) - set(test_fall_indices))
            test_dataset_no_misclassified = Subset(test_dataset, remaining_indices)
            dataloader_test_no_misclassified = DataLoader(test_dataset_no_misclassified, batch_size=16, shuffle=False)

            test_loss, test_accuracy, test_precision, test_recall, test_specificity, test_f1, true_labels, predictions = evaluate_model(model, dataloader_test_no_misclassified, criterion, device)

            logging_output(f"After Retraining - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test Specificity: {test_specificity:.4f}, Test F1-Score: {test_f1:.4f}", log_path)
                
            plot_confusion_matrix(true_labels, predictions, f'{experiment_name}_after_retraining', experiment_dir, classes=["No Fall", "Fall"])
    
    else:
        logging_output("No saved model found. Training a new model.", log_path)
        cross_validate_model(train_val_dataset, experiment_name, log_path, model_dir, experiment_dir)
    
        