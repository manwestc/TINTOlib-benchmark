from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import numpy as np
import torch
from scipy.special import softmax
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, average_precision_score, matthews_corrcoef, mean_squared_error,
    mean_absolute_error, r2_score
)

def get_loss_fn(task):

    if task == 'regression':
        # MSELoss expects:
        # - model output: (batch_size,) or (batch_size, 1)
        # - target:       same shape, dtype = float32
        return nn.MSELoss()
    elif task == 'binary':
        # BCEWithLogitsLoss expects:
        # - model output: (batch_size, 1) (logits, no sigmoid)
        # - target:       (batch_size, 1), dtype = float32
        return nn.BCEWithLogitsLoss()
        # CrossEntropyLoss expects:
        # - model output: (batch_size, num_classes) (logits, no softmax)
        # - target:       (batch_size,), dtype = long (class indices)
    elif task == 'multiclass':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task type: {task}")
    
def get_class_weighted_loss_fn(task, class_weight):

    if task == 'regression':
        return nn.MSELoss()  # no weighting for regression 
    elif task == 'binary':
        # Flatten and count positives/negatives
        return nn.BCEWithLogitsLoss(pos_weight=class_weight)
    elif task == 'multiclass':
        return nn.CrossEntropyLoss(weight=class_weight)
    else:
        raise ValueError(f"Unsupported task type: {task}")

def calculate_metrics(model, data_loader, device, class_weight=None, task='regression'):
    task = task.lower()
    model.eval()

    if task == 'regression':
        loss_fn = nn.MSELoss()  # no weighting for regression 
    elif task == 'binary':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)  # assumes model already applies sigmoid
    elif task == 'multiclass':
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    else:
        raise ValueError(f"Unsupported task: {task}")

    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for img_data, targets in data_loader:
            img_data = img_data.to(device)
            targets = targets.to(device)

            outputs = model(img_data)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_outputs)

    if task == 'regression':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {
            'loss': total_loss / len(data_loader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }, None, None, None

    elif task == 'binary':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        y_prob = 1 / (1 + np.exp(-y_pred)) 
        y_bin = (y_prob >= 0.5).astype(int)
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
            'f1': f1_score(y_true, y_bin, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'avg_precision': average_precision_score(y_true, y_prob),
            'mcc': matthews_corrcoef(y_true, y_bin)
        }, y_true, y_bin, y_prob

    elif task == 'multiclass':
        probs = softmax(y_pred, axis=1)
        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_pred_cls),
            'precision': precision_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_true, probs, multi_class='ovr', average='weighted'),
            'log_loss': log_loss(y_true, probs),
            'avg_precision': average_precision_score(y_true_bin, probs, average='weighted'),
            'mcc': matthews_corrcoef(y_true, y_pred_cls)
        }, y_true, y_pred_cls, probs
    
def calculate_metrics_hybrid(model, data_loader, device, class_weight=None, task='regression'):
    task = task.lower()
    model.eval()

    if task == 'regression':
        loss_fn = nn.MSELoss()  # no weighting for regression 
    elif task == 'binary':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)  # assumes model already applies sigmoid
    elif task == 'multiclass':
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    else:
        raise ValueError(f"Unsupported task: {task}")

    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for num_data, img_data, targets in data_loader:
            num_data = num_data.to(device)
            img_data = img_data.to(device)
            targets = targets.to(device)

            outputs = model(num_data, img_data)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_outputs)

    if task == 'regression':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {
            'loss': total_loss / len(data_loader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }, None, None, None

    elif task == 'binary':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        y_prob = 1 / (1 + np.exp(-y_pred)) 
        y_bin = (y_prob >= 0.5).astype(int)
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
            'f1': f1_score(y_true, y_bin, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'avg_precision': average_precision_score(y_true, y_prob),
            'mcc': matthews_corrcoef(y_true, y_bin)
        }, y_true, y_bin, y_prob

    elif task == 'multiclass':
        probs = softmax(y_pred, axis=1)
        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_pred_cls),
            'precision': precision_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_true, probs, multi_class='ovr', average='weighted'),
            'log_loss': log_loss(y_true, probs),
            'avg_precision': average_precision_score(y_true_bin, probs, average='weighted'),
            'mcc': matthews_corrcoef(y_true, y_pred_cls)
        }, y_true, y_pred_cls, probs
    
def calculate_metrics_manuel(model, data_loader, device, class_weight=None, task='regression', image_mode="left"):
    task = task.lower()
    model.eval()

    if task == 'regression':
        loss_fn = nn.MSELoss()  # no weighting for regression 
    elif task == 'binary':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)  # assumes model already applies sigmoid
    elif task == 'multiclass':
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    else:
        raise ValueError(f"Unsupported task: {task}")

    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for left_img_data, right_img_data, targets in data_loader:
            left_img_data, right_img_data, targets = left_img_data.to(device, non_blocking=True), right_img_data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            if image_mode == "left":
                outputs = model(left_img_data)
            elif image_mode == "right":
                outputs = model(right_img_data)
            elif image_mode == "both":
                outputs = model(left_img_data, right_img_data)
            else:
                raise ValueError(f"Invalid image_mode: {image_mode}")
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_outputs)

    if task == 'regression':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {
            'loss': total_loss / len(data_loader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }, None, None, None

    elif task == 'binary':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        y_prob = 1 / (1 + np.exp(-y_pred)) 
        y_bin = (y_prob >= 0.5).astype(int)
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
            'f1': f1_score(y_true, y_bin, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'avg_precision': average_precision_score(y_true, y_prob),
            'mcc': matthews_corrcoef(y_true, y_bin)
        }, y_true, y_bin, y_prob

    elif task == 'multiclass':
        probs = softmax(y_pred, axis=1)
        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_pred_cls),
            'precision': precision_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_true, probs, multi_class='ovr', average='weighted'),
            'log_loss': log_loss(y_true, probs),
            'avg_precision': average_precision_score(y_true_bin, probs, average='weighted'),
            'mcc': matthews_corrcoef(y_true, y_pred_cls)
        }, y_true, y_pred_cls, probs
    
def calculate_metrics_hybrid_manuel(model, data_loader, device, class_weight=None, task='regression', image_mode="left"):
    task = task.lower()
    model.eval()

    if task == 'regression':
        loss_fn = nn.MSELoss()  # no weighting for regression 
    elif task == 'binary':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)  # assumes model already applies sigmoid
    elif task == 'multiclass':
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    else:
        raise ValueError(f"Unsupported task: {task}")

    total_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for num_data, left_img_data, right_img_data, img_data, targets in data_loader:
            num_data, left_img_data, right_img_data, img_data, targets = num_data.to(device, non_blocking=True), left_img_data.to(device, non_blocking=True), right_img_data.to(device, non_blocking=True), img_data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            if image_mode == "left":
                outputs = model(num_data, left_img_data)
            elif image_mode == "right":
                outputs = model(num_data, right_img_data)
            elif image_mode == "both":
                outputs = model(num_data, left_img_data, right_img_data)
            elif image_mode == "refined_only":
                outputs = model(img_data, img_data)
            elif image_mode == "refined_both":
                outputs = model(num_data, left_img_data, right_img_data, img_data)
            else:
                raise ValueError(f"Invalid image_mode: {image_mode}")
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_outputs)

    if task == 'regression':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {
            'loss': total_loss / len(data_loader),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }, None, None, None

    elif task == 'binary':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        y_prob = 1 / (1 + np.exp(-y_pred)) 
        y_bin = (y_prob >= 0.5).astype(int)
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
            'f1': f1_score(y_true, y_bin, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'avg_precision': average_precision_score(y_true, y_prob),
            'mcc': matthews_corrcoef(y_true, y_bin)
        }, y_true, y_bin, y_prob

    elif task == 'multiclass':
        probs = softmax(y_pred, axis=1)
        y_pred_cls = np.argmax(y_pred, axis=1)
        y_true_bin = label_binarize(y_true, classes=np.arange(probs.shape[1]))
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(y_true, y_pred_cls),
            'precision': precision_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_true, probs, multi_class='ovr', average='weighted'),
            'log_loss': log_loss(y_true, probs),
            'avg_precision': average_precision_score(y_true_bin, probs, average='weighted'),
            'mcc': matthews_corrcoef(y_true, y_pred_cls)
        }, y_true, y_pred_cls, probs

def calculate_metrics_from_numpy(y_true, y_pred, task):
    if task == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    elif task == 'binary':
        y_bin = (y_pred >= 0.5).astype(int)
        return {
            'accuracy': accuracy_score(y_true, y_bin),
            'precision': precision_score(y_true, y_bin, zero_division=0),
            'recall': recall_score(y_true, y_bin, zero_division=0),
            'f1': f1_score(y_true, y_bin, zero_division=0)
        }
    elif task == 'multiclass':
        y_pred_cls = np.argmax(y_pred, axis=1)
        return {
            'accuracy': accuracy_score(y_true, y_pred_cls),
            'precision': precision_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred_cls, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred_cls, average='weighted', zero_division=0)
        }
    else:
        raise ValueError(f'Unsupported task: {task}')
