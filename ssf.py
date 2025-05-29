import torch
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils import *

import argparse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='unsw')
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--epoch_1", type=int, default=1)
parser.add_argument("--percent", type=float, default=0.8)
parser.add_argument("--sample_interval", type=int, default=20000)
parser.add_argument("--cuda", type=str, default="2")
parser.add_argument("--num_labeled_sample", type=int, default=200)
parser.add_argument("--opt_new_lr", type=float, default=50.0)
parser.add_argument("--opt_old_lr", type=float, default=1.0)
parser.add_argument("--new_sample_weight", type=float, default=100.0, help="Weight for new samples in the loss calculation")


args = parser.parse_args()
dataset = args.dataset
epochs = args.epochs
epoch_1 = args.epoch_1
percent = args.percent
sample_interval = args.sample_interval
cuda_num = args.cuda
num_labeled_sample = args.num_labeled_sample
opt_new_lr = args.opt_new_lr
opt_old_lr = args.opt_old_lr
new_sample_weight = args.new_sample_weight

seed = 5011
seed_round = 5
old_init = '0.5-1'
new_init = '0-0.5'
lwf_lambda = 0.5

tem = 0.02
bs = 128
drift_threshold = 0.05

if dataset == 'nsl':
    input_dim = 121
else:
    input_dim = 196


if dataset == 'nsl':
    KDDTrain_dataset_path   = "NSL_pre_data/PKDDTrain+.csv"
    KDDTest_dataset_path    = "NSL_pre_data/PKDDTest+.csv"

    KDDTrain   =  load_data(KDDTrain_dataset_path)
    KDDTest    =  load_data(KDDTest_dataset_path)

    splitter_nsl = SplitData(dataset='nsl')
else:
    UNSWTrain_dataset_path   = "UNSW_pre_data/UNSWTrain.csv"
    UNSWTest_dataset_path    = "UNSW_pre_data/UNSWTest.csv"

    UNSWTrain   =  load_data(UNSWTrain_dataset_path)
    UNSWTest    =  load_data(UNSWTest_dataset_path)

    splitter_unsw = SplitData(dataset='unsw')

device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")

criterion = InfoNCELoss(device, tem)
if dataset != 'nsl':
    classification_criterion = nn.BCELoss(reduction='none')

for i in range(seed_round):
    # Set the seed for the random number generator for this iteration
    # setup_seed(seed+i)
    current_seed = seed+i
    setup_seed(current_seed)
    print(f"Current seed: {current_seed}")

    if dataset == 'nsl':
        # Transform the data
        x_train, y_train = splitter_nsl.transform(KDDTrain, labels='labels2')
        x_test, y_test = splitter_nsl.transform(KDDTest, labels='labels2')
    else:
        # Transform the data
        x_train, y_train = splitter_unsw.transform(UNSWTrain, labels='label')
        x_test, y_test = splitter_unsw.transform(UNSWTest, labels='label')

    # Convert to torch tensors
    x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
    x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)
    print('shape of x_train ', x_train.shape)
    print('shape of x_test is ', x_test.shape)

    torch.cuda.empty_cache()

    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(x_train, y_train, test_size=percent)
    memory = x_train.shape[0] * (1-percent)
    print('size of memory is ', memory)
    memory = math.floor(memory)

    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=bs, shuffle=True)
    
    num_of_first_train = online_x_train.shape[0]

    if dataset == 'nsl':
        model = AE(input_dim).to(device)
        teacher_model = AE(input_dim).to(device)
    else:
        model = AE_classifier(input_dim).to(device)
        teacher_model = AE_classifier(input_dim).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

    model.train()
    for epoch in range(epochs):
        if epoch % 50 == 0 or epoch == 0:
            print('seed = ', (seed+i), ', first round: epoch = ', epoch)
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)

            labels = labels.to(device)
            optimizer.zero_grad()

            if dataset == 'nsl':
                features, recon_vec = model(inputs)
            else:
                features, recon_vec, classifications = model(inputs)
            con_loss = criterion(recon_vec,labels)
            if dataset == 'nsl':
                loss = con_loss.mean()
            else:
                classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                loss = con_loss.mean() + classification_loss.mean()

            loss.backward()
            optimizer.step()

    teacher_model.load_state_dict(model.state_dict())  # Initialize teacher model

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    online_x_train, online_y_train  = online_x_train.to(device), online_y_train.to(device)

    x_train_this_epoch, x_test_left_epoch, y_train_this_epoch, y_test_left_epoch = online_x_train.clone(), online_x_test.clone().to(device), online_y_train.clone(), online_y_test.clone()
    
    permutation = torch.randperm(x_test.size(0))

    # Apply this permutation to both tensors to shuffle them in unison.
    x_test = x_test[permutation]
    y_test = y_test[permutation]
    # Concatenating x_test and x_test_left_epoch
    x_test_left_epoch = torch.cat((x_test_left_epoch, x_test), dim=0)
    print('shape of x_test_left_epoch is ', x_test_left_epoch.shape)
    # Concatenating y_test and y_test_left_epoch
    y_test_left_epoch = torch.cat((y_test_left_epoch, y_test), dim=0)

    if dataset == 'nsl':
        normal_recon_temp = torch.mean(F.normalize(model(online_x_train[(online_y_train == 0).squeeze()])[1], p=2, dim=1), dim=0)
        y_pred_before_online = evaluate(normal_recon_temp, online_x_train, online_y_train, x_test_left_epoch, 0, model)
        y_pred_before_online = y_pred_before_online.cpu().numpy()
        print('--------------------------- performance_before_continual_training -----------------------------------')
        performance_before = score_detail(y_test_left_epoch[-x_test.shape[0]:].numpy(), y_pred_before_online[-x_test.shape[0]:].astype("int32"))
    else:
        print('--------------------------- performance_before_contunual_training -----------------------------------')
        test_ds = TensorDataset(x_test, y_test)  # Replace with your test data
        test_loader = DataLoader(dataset=test_ds, batch_size=bs, shuffle=False)
        performance_before = evaluate_classifier(model, test_loader, device)

####################### start online training #######################
    model = model.to(device)

    count = 0
    y_train_detection = y_train_this_epoch

    start_idx = 0
    y_train_detection = torch.empty(0, dtype=torch.long, device=device)
    labeled_indices = []

    while start_idx < len(x_test_left_epoch):
        print('seed = ', (seed+i), ', i = ', count)
        count += 1
        
        end_idx = min(start_idx + sample_interval, len(x_test_left_epoch))
        x_test_this_epoch = x_test_left_epoch[start_idx:end_idx]
        y_test_this_epoch = y_test_left_epoch[start_idx:end_idx]

        model = model.to(device)

        start_idx += sample_interval

        if dataset == 'nsl':
            # must compute the normal_temp and normal_recon_temp again, because the model has been updated
            normal_recon_temp = torch.mean(F.normalize(model(x_train_this_epoch[(y_train_this_epoch == 0).squeeze()])[1], p=2, dim=1), dim=0)
            pdf1, pdf2, values1 = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, x_train_this_epoch, y_train_this_epoch, model, get_probs=True)
            pdf11, pdf22, values11 = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, 0, model, get_probs=True)
            
            pdf1_probe = pdf1 / (pdf1 + pdf2)
            pdf11_probe = pdf11 / (pdf11 + pdf22)

            drift = detect_drift(pdf11_probe, pdf1_probe, sample_interval, drift_threshold)
            
            control_res = pdf1_probe.cpu().numpy()
            treatment_res = pdf11_probe.cpu().numpy()
        else:
            # Get logits for drift detection
            with torch.no_grad():
                _, _, test_logits = model(x_test_this_epoch)
                test_logits = test_logits.squeeze()

            with torch.no_grad():
                _, _, train_logits = model(x_train_this_epoch)
                train_logits = train_logits.squeeze()
            
            drift = detect_drift(test_logits, train_logits, sample_interval, drift_threshold)
            control_res = train_logits.cpu().numpy()
            treatment_res = test_logits.cpu().numpy()

        # Optimize masks for old and new data
        M_c = optimize_old_mask(control_res, treatment_res, device, initialization=old_init, lr=opt_old_lr)
        M_t = optimize_new_mask(control_res, treatment_res, M_c, device, initialization=new_init, lr=opt_new_lr)

        y_test_this_epoch = y_test_this_epoch.to(device)

        # Detect Drift
        if drift:
            print("Drift detected, update the model...")
            # Select and update representative samples
            if dataset == 'nsl':
                x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples_when_drift(
                    x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device, memory, model, normal_recon_temp
                )
            else:
                x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples_when_drift(
                    x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device, memory, model
                )
        else:
            # Select and update representative samples
            x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples(
                x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, y_test_this_epoch, M_c, M_t, num_labeled_sample, device
            )

        labeled_indices.append(start_idx - sample_interval + labeled_indices_current.cpu().numpy())

        # Re-training model
        train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch, new_mask)
        train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=bs, shuffle=True)
        
        model = model.to(device)
        model.train()

        if drift:            
            for epoch in range(epoch_1):
                if epoch % 50 == 0:
                    print('epoch = ', epoch)
                for j, data in enumerate(train_loader, 0):
                    inputs, labels, new_sample_mask = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    new_sample_mask = new_sample_mask.to(device)
                    normal_new_mask = new_sample_mask[labels == 0]
                    
                    optimizer.zero_grad()
                    
                    if dataset == 'nsl':
                        features, recon_vec = model(inputs)
                    else:
                        features, recon_vec, classifications = model(inputs)

                    con_loss = criterion(recon_vec, labels)
                    weighted_con_loss = con_loss * ((1 - normal_new_mask) + normal_new_mask * new_sample_weight)

                    if dataset == 'nsl':
                        weighted_loss = weighted_con_loss.mean()
                    else:
                        classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                        weighted_classification_loss = classification_loss * ((1 - new_sample_mask) + new_sample_mask * new_sample_weight)
                        weighted_loss = weighted_con_loss.mean() + weighted_classification_loss.mean()

                    weighted_loss.backward()
                    optimizer.step()
        else:      
            for epoch in range(epoch_1):
                if epoch % 50 == 0:
                    print('epoch = ', epoch)
                for j, data in enumerate(train_loader, 0):
                    inputs, labels, new_sample_mask = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    new_sample_mask = new_sample_mask.to(device)
                    normal_new_mask = new_sample_mask[labels == 0]
                    
                    optimizer.zero_grad()
                    
                    if dataset == 'nsl':
                        features, recon_vec = model(inputs)
                    else:
                        features, recon_vec, classifications = model(inputs)

                    con_loss = criterion(recon_vec, labels)
                    weighted_con_loss = con_loss * ((1 - normal_new_mask) + normal_new_mask * new_sample_weight)

                    if dataset == 'nsl':
                        weighted_loss = weighted_con_loss.mean()
                    else:
                        classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                        weighted_classification_loss = classification_loss * ((1 - new_sample_mask) + new_sample_mask * new_sample_weight)
                        weighted_loss = weighted_con_loss.mean() + weighted_classification_loss.mean()

                    # Apply the new sample weight using the mask
                    # weighted_loss = (1 - mask) * loss + mask * new_sample_weight * loss

                    if dataset == 'nsl':
                        with torch.no_grad():
                            teacher_features, teacher_recon_vec = teacher_model(inputs)
                        distillation_loss = F.mse_loss(recon_vec, teacher_recon_vec)
                    else:
                        with torch.no_grad():
                            teacher_features, teacher_recon_vec, teacher_logits = teacher_model(inputs)
                        distillation_loss = F.mse_loss(classifications, teacher_logits)
                    total_loss = weighted_loss + lwf_lambda * distillation_loss

                    total_loss.backward()
                    optimizer.step()
        
        teacher_model.load_state_dict(model.state_dict())  # Update teacher model

        if dataset == 'nsl':
            # normal_temp = torch.mean(F.normalize(model(x_train_this_epoch[(y_train_this_epoch == 0).squeeze()])[0], p=2, dim=1), dim=0)
            normal_recon_temp = torch.mean(F.normalize(model(x_train_this_epoch[(y_train_this_epoch == 0).squeeze()])[1], p=2, dim=1), dim=0)
            predict_label = evaluate(normal_recon_temp, x_train_this_epoch, y_train_this_epoch, x_test_this_epoch, 0, model)
        else:
            test_ds = TensorDataset(x_test_this_epoch, y_test_this_epoch)  # Replace with your test data
            test_loader = DataLoader(dataset=test_ds, batch_size=bs, shuffle=False)
            predict_label = evaluate_classifier(model, test_loader, device, get_predict=True)
        
        y_train_detection = torch.cat((y_train_detection.to(device), torch.tensor(predict_label).to(device)))

################### test the performance after online training ###################

    test_size = len(x_test)
    total_size = len(x_test_left_epoch)

    # Convert list of numpy arrays to a single numpy array
    all_labeled_indices = np.hstack(labeled_indices)

    # Mask to denote indexes with true label
    mask = np.ones(total_size, dtype=bool)
    mask[all_labeled_indices] = False

    # get the mask to denote indexes that have been true-labeled in the test dataset
    test_mask = mask[-test_size:]

    # obtain true and pseudo labels in test dataset
    y_test_left_pseudo = y_train_detection[-test_size:][test_mask].to(device)
    y_test_left_true = y_test_left_epoch[-test_size:][test_mask].to(device)
    
    print('--------------------------- performance_after_continual_training -----------------------------------')
    performance_test = score_detail(y_test_left_true.cpu().numpy(), y_test_left_pseudo.cpu().numpy())



