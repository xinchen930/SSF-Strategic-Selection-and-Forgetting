import os
import torch
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.metrics import accuracy_score
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from scipy.stats import ks_2samp

import torch.optim as optim
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
    
# MLP function for unsw
class AE_classifier(nn.Module):
    def __init__(self, input_dim):
        super(AE_classifier, self).__init__()
        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 1),  # 1 neuron for binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        classify = self.classifier(decode)
        return encode, decode, classify

# Evaluation function for unsw
def evaluate_classifier(model, data_loader, device, get_predict=False):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            _, _, classifications = model(inputs)
            preds = (classifications.squeeze() > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    if not get_predict:
        res = score_detail(all_labels, all_preds, if_print=True)

    if get_predict:
        return all_preds
    else:
        return res

# Evaluation function for single sample or batch of samples for unsw
def evaluate_inputs(model, inputs, device):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        _, _, classifications = model(inputs)
        preds = (classifications.squeeze() > 0.5).float()
    return preds.cpu().numpy()

def initialize_tensor(size, initialization, device):
    if initialization == '0-1':
        return torch.nn.Parameter(torch.rand(size, device=device), requires_grad=True)
    elif initialization == '0-0.5':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5, requires_grad=True)
    elif initialization == '0.5-1':
        return torch.nn.Parameter(torch.rand(size, device=device) * 0.5 + 0.5, requires_grad=True)
    else:
        raise ValueError("Invalid initialization type. Choose from '0-1', '0-0.5', or '0.5-1'.")

def optimize_old_mask(control_res, treatment_res, device, initialization='0-1', num_bins=10, lr=100, steps=100):
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)

    M_c = initialize_tensor(control_res.size(0), initialization, device)

    optimizer = torch.optim.SGD([M_c], lr=lr)
    delta = 1e-4

    for step in range(steps):
        with torch.no_grad():
            M_c.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        control_hist = torch.histc(control_res, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_obs_c = torch.zeros(num_bins, device=device)
        bin_tgt_c = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            bin_obs_c[i] = torch.sum(M_c * mask_c.float()) / torch.sum(M_c)
            bin_tgt_c[i] = treatment_hist[i] / len(treatment_res)

        bin_obs_c = bin_obs_c / bin_obs_c.sum()
        bin_tgt_c = bin_tgt_c / bin_tgt_c.sum()

        Accuracy_Loss_c = F.kl_div(bin_obs_c.log(), bin_tgt_c, reduction='sum')

        Loss = Accuracy_Loss_c
        Loss.backward()
        optimizer.step()

    return M_c

def optimize_new_mask(control_res, treatment_res, M_c, device, initialization='0-1', num_bins=10, lr=0.1, steps=100):
    control_res = torch.tensor(control_res, dtype=torch.float).to(device)
    treatment_res = torch.tensor(treatment_res, dtype=torch.float).to(device)

    M_t = initialize_tensor(treatment_res.size(0), initialization, device)

    optimizer = torch.optim.SGD([M_t], lr=lr)
    delta = 1e-4

    for step in range(steps):
        with torch.no_grad():
            M_t.clamp_(delta, 1 - delta)

        optimizer.zero_grad()

        bin_edges = torch.linspace(0., 1., num_bins + 1, device=device)
        Drift_Loss_t = 0.

        control_hist = torch.histc(control_res, bins=num_bins, min=0., max=1.)
        treatment_hist = torch.histc(treatment_res, bins=num_bins, min=0., max=1.)

        bin_tgt_t = torch.zeros(num_bins, device=device)

        bin_combined = torch.zeros(num_bins, device=device)
        for i in range(num_bins):
            mask_c = (control_res >= bin_edges[i]) & (control_res < bin_edges[i + 1])
            mask_t = (treatment_res >= bin_edges[i]) & (treatment_res < bin_edges[i + 1])

            bin_tgt_t[i] = treatment_hist[i] / len(treatment_res)

            bin_combined[i] = (torch.sum(M_t * mask_t.float()) + torch.sum(M_c * mask_c.float())) / (torch.sum(M_t) + torch.sum(M_c))

        bin_combined = torch.clamp(bin_combined / bin_combined.sum(), min=1e-10)
        bin_combined = bin_combined / bin_combined.sum()
        bin_tgt_t = torch.clamp(bin_tgt_t / bin_tgt_t.sum(), min=1e-10)
        bin_tgt_t = bin_tgt_t / bin_tgt_t.sum()

        Drift_Loss_t = F.kl_div(bin_combined.log(), bin_tgt_t, reduction='sum')

        Loss = Drift_Loss_t
        Loss.backward()
        optimizer.step()

    return M_t

def select_and_update_representative_samples(x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device):
    M_c_bin = (M_c >= 0.5).float().to(device)
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    print(f"Selected representative old samples: {representative_old.shape}")
    print(f"Selected representative new samples: {representative_new.shape}")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    if len(non_representative_old_indices) < num_to_remove:
        print(f"Not enough non-representative old samples to remove ({len(non_representative_old_indices)}). Removing additional representative samples.")
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        
        # Remove all non-representative samples first
        remove_indices = non_representative_old_indices
        
        # Then remove the remaining number from the representative samples with the lowest scores
        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])
    else:
        remove_indices = non_representative_old_indices[torch.randperm(len(non_representative_old_indices))[:num_to_remove]]

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        print(f"Not enough representative new samples selected ({representative_new.shape[0]}). Selecting additional random samples.")
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat([x_train_this_epoch, drift_representative_new], dim=0)
    y_train_this_epoch = torch.cat([y_train_this_epoch, new_labels], dim=0)

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

def select_and_update_representative_samples_when_drift(
        x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, 
        M_c, M_t, num_labeled_sample, device, buffer_memory_size, model, normal_recon_temp=None):

    M_c_bin = (M_c >= 0.5).float().to(device)
    M_t_bin = (M_t >= 0.5).float().to(device)

    representative_old = x_train_this_epoch[M_c_bin.bool()]
    representative_new = x_test_this_epoch[M_t_bin.bool()]

    print(f"Selected representative old samples: {representative_old.shape}")
    print(f"Selected representative new samples: {representative_new.shape}")

    old_indices = torch.arange(len(x_train_this_epoch), device=device)
    representative_old_indices = old_indices[M_c_bin.bool()]

    mask_c = torch.ones(len(x_train_this_epoch), dtype=torch.bool, device=device)
    mask_c[representative_old_indices] = False

    non_representative_old_indices = old_indices[mask_c]
    num_to_remove = num_labeled_sample

    # Remove all non-representative samples
    remove_indices = non_representative_old_indices

    if len(non_representative_old_indices) < num_to_remove:
        print(f"Not enough non-representative old samples to remove ({len(non_representative_old_indices)}). Removing additional representative samples.")
        additional_remove_needed = num_to_remove - len(non_representative_old_indices)
        
        # Then remove the remaining number from the representative samples with the lowest scores
        representative_scores = M_c[M_c_bin.bool()].detach().cpu().numpy()
        sorted_rep_indices = torch.argsort(torch.tensor(representative_scores))[:additional_remove_needed]
        additional_remove_indices = representative_old_indices[sorted_rep_indices]

        remove_indices = torch.cat([remove_indices, additional_remove_indices])

    mask = torch.ones(x_train_this_epoch.size(0), dtype=torch.bool, device=device)
    mask[remove_indices] = False

    x_train_this_epoch = x_train_this_epoch[mask]
    y_train_this_epoch = y_train_this_epoch[mask]

    new_sample_mask = torch.zeros_like(y_train_this_epoch, dtype=torch.float32).to(device)

    if representative_new.shape[0] < num_labeled_sample:
        print(f"Not enough representative samples selected ({representative_new.shape[0]}). Selecting additional random samples.")
        additional_samples_needed = num_labeled_sample - representative_new.shape[0]

        selected_indices = set(torch.arange(len(x_test_this_epoch))[M_t_bin.bool().cpu().numpy()])
        available_indices = set(torch.arange(len(x_test_this_epoch)).cpu().numpy()) - selected_indices
        available_indices = torch.tensor(list(available_indices), dtype=torch.long)

        fallback_indices = available_indices[torch.randperm(len(available_indices))[:additional_samples_needed]]
        drift_representative_new = torch.cat([representative_new, x_test_this_epoch[fallback_indices]], dim=0)
        new_labels = torch.cat([y_test_this_epoch[M_t_bin.bool()], y_test_this_epoch[fallback_indices]], dim=0)
        sorted_indices_new = torch.cat([torch.arange(len(representative_new)), fallback_indices], dim=0)
    else:
        scores_new = M_t[M_t_bin.bool()].detach().cpu().numpy()
        sorted_indices_new = torch.argsort(torch.tensor(scores_new), descending=True)[:num_labeled_sample]
        drift_representative_new = representative_new[sorted_indices_new]
        new_labels = y_test_this_epoch[M_t_bin.bool()][sorted_indices_new]

    new_sample_mask = torch.cat([new_sample_mask, torch.ones(len(drift_representative_new), dtype=torch.float32).to(device)])
    x_train_this_epoch = torch.cat((x_train_this_epoch, drift_representative_new), dim=0)
    y_train_this_epoch = torch.cat((y_train_this_epoch, new_labels), dim=0)

    if len(x_train_this_epoch) < buffer_memory_size:
        additional_samples_needed = buffer_memory_size - len(x_train_this_epoch)
        print(f"Buffer memory has extra space for {additional_samples_needed} samples. Adding new samples with pseudo labels.")

        if representative_new.shape[0] > num_labeled_sample:
            remaining_new_samples = representative_new[torch.argsort(torch.tensor(scores_new), descending=True)[num_labeled_sample:]]
            # remaining_samples_needed = num_additional_samples_needed

            if remaining_new_samples.size(0) >= additional_samples_needed:
                pseudo_labeled_samples = remaining_new_samples[:additional_samples_needed]
                if normal_recon_temp == None:
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
                else:
                    pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, pseudo_labeled_samples, 0, model)
            else:
                pseudo_labeled_samples = remaining_new_samples
                if normal_recon_temp == None:
                    pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
                else:
                    pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, pseudo_labeled_samples, 0, model)
                
                random_new_additional_samples_needed = additional_samples_needed - remaining_new_samples.size(0)
                
                additional_indices = torch.randperm(len(x_test_this_epoch))[:random_new_additional_samples_needed]
                additional_pseudo_labeled_samples = x_test_this_epoch[additional_indices]
                if normal_recon_temp == None:
                    additional_pseudo_labels = evaluate_inputs(model, additional_pseudo_labeled_samples, device)
                else:
                    additional_pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, additional_pseudo_labeled_samples, 0, model)
                
                pseudo_labeled_samples = torch.cat([pseudo_labeled_samples, additional_pseudo_labeled_samples], dim=0)
                if normal_recon_temp == None:
                    if random_new_additional_samples_needed > 1:
                        pseudo_labels = torch.cat([torch.tensor(pseudo_labels), torch.tensor(additional_pseudo_labels)], dim=0)
                    else:
                        pseudo_labels = torch.cat([torch.tensor(pseudo_labels), torch.tensor(additional_pseudo_labels).unsqueeze(0)], dim=0)
                else:
                    if random_new_additional_samples_needed > 1:
                        pseudo_labels = torch.cat([pseudo_labels, additional_pseudo_labels], dim=0)
                    else:
                        pseudo_labels = torch.cat([pseudo_labels, additional_pseudo_labels.unsqueeze(0)], dim=0)
        else:
            additional_indices = torch.randperm(len(x_test_this_epoch))[:additional_samples_needed]
            pseudo_labeled_samples = x_test_this_epoch[additional_indices]
            if normal_recon_temp == None:
                pseudo_labels = evaluate_inputs(model, pseudo_labeled_samples, device)
            else:
                pseudo_labels = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, pseudo_labeled_samples, 0, model)

        x_train_this_epoch = torch.cat((x_train_this_epoch, pseudo_labeled_samples), dim=0)
        if normal_recon_temp == None:
            if additional_samples_needed > 1:
                y_train_this_epoch = torch.cat((y_train_this_epoch, torch.tensor(pseudo_labels).to(device)), dim=0)
            else:
                y_train_this_epoch = torch.cat((y_train_this_epoch, torch.tensor(pseudo_labels).unsqueeze(0).to(device)), dim=0)
        else:
            if additional_samples_needed > 1:
                y_train_this_epoch = torch.cat((y_train_this_epoch, pseudo_labels.to(device)), dim=0)
            else:
                y_train_this_epoch = torch.cat((y_train_this_epoch, pseudo_labels.unsqueeze(0).to(device)), dim=0)
        
        new_sample_mask = torch.cat([new_sample_mask, torch.zeros(len(pseudo_labeled_samples), dtype=torch.float32).to(device)])

    return x_train_this_epoch, y_train_this_epoch, sorted_indices_new, new_sample_mask

#################################################################

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super(SplitData, self).__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            # Preparing the labels
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            # abnormal data is labeled as 1, normal data 0
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')

        elif self.dataset == 'unsw':
            # UNSW dataset processing
            y_ = X[labels]
            X_ = X.drop('label', axis=1)

        else:
            raise ValueError("Unsupported dataset type")

        # Normalization
        normalize = MinMaxScaler().fit(X_)
        x_ = normalize.transform(X_)

        return x_, y_

class AE(nn.Module):
    def __init__(self, input_dim):
        super(AE, self).__init__()

        # Find the nearest power of 2 to input_dim
        nearest_power_of_2 = 2 ** round(math.log2(input_dim))

        # Calculate the dimensions of the 2nd/4th layer and the 3rd layer.
        second_fourth_layer_size = nearest_power_of_2 // 2  # A half
        third_layer_size = nearest_power_of_2 // 4         # A quarter

        # Create encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, third_layer_size),
        )

        # Create decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(third_layer_size, second_fourth_layer_size),
            nn.ReLU(),
            nn.Linear(second_fourth_layer_size, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class InfoNCELoss(nn.Module):
    def __init__(self, device, temperature=0.1, scale_by_temperature=True):
        super(InfoNCELoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()
        # compute logits
        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # Calculate the dot product similarity between pairwise samples
        # create mask 
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size).to(self.device)  
        logits_without_ii = logits * logits_mask
        
        logits_normal = logits_without_ii[(labels == 0).squeeze()]
        logits_normal_normal = logits_normal[:,(labels == 0).squeeze()]
        logits_normal_abnormal = logits_normal[:,(labels > 0).squeeze()]

        sum_of_vium = torch.sum(torch.exp(logits_normal_abnormal), axis=1, keepdims=True)
        denominator = torch.exp(logits_normal_normal) + sum_of_vium
        log_probs = logits_normal_normal - torch.log(denominator)
  
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature

        return loss
    
def score_detail(y_test,y_test_pred,if_print=True):
    # Confusion matrix
    print("Confusion matrix")
    print(confusion_matrix(y_test, y_test_pred))
    # Accuracy 
    print('Accuracy ',accuracy_score(y_test, y_test_pred))
    # Precision 
    print('Precision ',precision_score(y_test, y_test_pred))
    # Recall
    print('Recall ',recall_score(y_test, y_test_pred))
    # F1 score
    print('F1 score ',f1_score(y_test,y_test_pred))

    return accuracy_score(y_test, y_test_pred), precision_score(y_test, y_test_pred), recall_score(y_test, y_test_pred), f1_score(y_test,y_test_pred)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# Define a small epsilon to avoid log(0)
EPSILON = 1e-10

def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2 = params
    pdf1 = gaussian_pdf(data, mu1, sigma1)
    pdf2 = gaussian_pdf(data, mu2, sigma2)
    
    # Ensure the values passed to log are above a threshold
    likelihood = 0.5 * pdf1 + 0.5 * pdf2
    likelihood = np.clip(likelihood, a_min=EPSILON, a_max=None)
    
    # Check for NaN values
    if np.any(np.isnan(likelihood)):
        print("NaN values found in likelihood calculation")
        return np.inf
    
    return -np.sum(np.log(likelihood))

# Define the batch processing function
def process_batch(data, temp, layer_index, model, batch_size=128, device='cuda'):
    values = []
    model.to(device)
    temp = temp.to(device)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        batch_features = F.normalize(model(batch)[layer_index], p=2, dim=1)
        batch_cosine_sim = F.cosine_similarity(batch_features, temp.reshape([-1, temp.shape[0]]), dim=1)
        values.append(batch_cosine_sim)
        del batch, batch_features, batch_cosine_sim  # Free memory
        torch.cuda.empty_cache()  # Clear cache
    
    return torch.cat(values)

def evaluate(normal_recon_temp, x_train, y_train, x_test, y_test, model, batch_size=128, device='cuda', get_probs=False):
    model.eval()
    # Define dataset and dataloader
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)

    # num_of_layer = 0
    num_of_output = 1

    # values_features_all = []
    # values_features_normal = []
    # values_features_abnormal = []
    values_recon_all = []
    values_recon_normal = []
    values_recon_abnormal = []

    model.to(device)
    # normal_temp = normal_temp.to(device)
    normal_recon_temp = normal_recon_temp.to(device)

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            features, recon_vec = model(inputs)
            # values_features_all.append(F.cosine_similarity(F.normalize(features, p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
            values_recon_all.append(F.cosine_similarity(F.normalize(recon_vec, p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

            normal_mask = (labels == 0)
            abnormal_mask = (labels == 1)

            if normal_mask.sum() > 0:
                # values_features_normal.append(F.cosine_similarity(F.normalize(features[normal_mask], p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
                values_recon_normal.append(F.cosine_similarity(F.normalize(recon_vec[normal_mask], p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))
            
            if abnormal_mask.sum() > 0:
                # values_features_abnormal.append(F.cosine_similarity(F.normalize(features[abnormal_mask], p=2, dim=1), normal_temp.reshape([-1, normal_temp.shape[0]]), dim=1))
                values_recon_abnormal.append(F.cosine_similarity(F.normalize(recon_vec[abnormal_mask], p=2, dim=1), normal_recon_temp.reshape([-1, normal_recon_temp.shape[0]]), dim=1))

        values_recon_all = torch.cat(values_recon_all).cpu().numpy()
        # values_recon_all = torch.cat(values_recon_all)
        values_recon_normal = torch.cat(values_recon_normal).cpu().numpy()
        values_recon_abnormal = torch.cat(values_recon_abnormal).cpu().numpy()

    x_test = x_test.to(device)
    # values_features_test = process_batch(x_test, normal_temp, num_of_layer, model, batch_size, device)
    values_recon_test = process_batch(x_test, normal_recon_temp, num_of_output, model, batch_size, device)

    mu3_initial = np.mean(values_recon_normal)
    sigma3_initial = np.std(values_recon_normal)
    mu4_initial = np.mean(values_recon_abnormal)
    sigma4_initial = np.std(values_recon_abnormal)

    # Fit Gaussians to reconstruction similarities
    initial_params = np.array([mu3_initial, sigma3_initial, mu4_initial, sigma4_initial])
    result = opt.minimize(log_likelihood, initial_params, args=(values_recon_all,), method='Nelder-Mead')
    mu3_fit, sigma3_fit, mu4_fit, sigma4_fit = result.x

    if mu3_fit > mu4_fit:
        gaussian3 = Normal(mu3_fit, sigma3_fit)
        gaussian4 = Normal(mu4_fit, sigma4_fit)
    else:
        gaussian4 = Normal(mu3_fit, sigma3_fit)
        gaussian3 = Normal(mu4_fit, sigma4_fit)

    pdf3 = gaussian3.log_prob(values_recon_test.clone().detach()).exp()
    pdf4 = gaussian4.log_prob(values_recon_test.clone().detach()).exp()
    y_test_pred_4 = (pdf4 > pdf3).cpu().numpy().astype("int32")
    # y_test_pro_de = (torch.abs(pdf4 - pdf3)).cpu().detach().numpy().astype("float32")

    if get_probs:
        values_recon_test = values_recon_test.detach()
        return pdf3,pdf4,values_recon_test
    else:
        if not isinstance(y_test, int):
            if y_test.device != torch.device("cpu"):
                y_test = y_test.cpu().numpy()
            result_decoder = score_detail(y_test, y_test_pred_4)

        # y_test_pred_no_vote = torch.where(torch.from_numpy(y_test_pro_en) > torch.from_numpy(y_test_pro_de), torch.from_numpy(y_test_pred_2), torch.from_numpy(y_test_pred_4))
        y_test_pred_no_vote = torch.from_numpy(y_test_pred_4)

        if not isinstance(y_test, int):
            result_final = score_detail(y_test, y_test_pred_no_vote, if_print=True)
            # return result_encoder, result_decoder, result_final
            return result_decoder, result_final
        else:
            return y_test_pred_no_vote

# drift detection
def detect_drift(new_data, control_data, window_size, drift_threshold):
    for i in range(0, len(new_data), window_size):
        window_data = new_data[i:i + window_size]
        if len(window_data) < window_size:
            break
        ks_statistic, p_value = ks_2samp(control_data.cpu().numpy(), window_data.cpu().numpy())
        # ks_statistic, p_value = ks_2samp(control_data, window_data)
        if p_value < drift_threshold:
            print(f"!!!!!!!!!!!!!!!!!!!!! Drift detected in window {i // window_size + 1} (p-value: {p_value})")
            return True
        else:
            print(f"No drift detected in window {i // window_size + 1} (p-value: {p_value})")
    return False
