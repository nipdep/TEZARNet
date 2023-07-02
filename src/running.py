from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from .utils.analysis import action_evaluator


def train_step(model, dataloader, optimizer, loss_module, loss_alpha, device, class_names, target_feat_met, phase='train', l2_reg=False, with_attr=False):
    model = model.train()

    epoch_loss = 0  # total loss of epoch
    total_samples = 0  # total samples in epoch

    with tqdm(dataloader, unit="batch", desc=phase) as tepoch:
        for batch in tepoch:
            X, targets, target_feat, target_attr, padding_masks = batch
            # print(X, targets, target_feat, target_attr)
            X = X.float().to(device)
            target_feat = target_feat.float().to(device)
            target_attr = target_attr.float().to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)  # 0s: ignore

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            # with autocast():
                if with_attr:
                    attr_output, feat_output = model(X)
                else:
                    feat_output = model(X)
                # print("feature shape", feat_output.shape, "target feature shape", target_feat.shape)
                feat_loss = loss_module['feature'](feat_output, target_feat)
                if with_attr:
                    attr_loss = loss_module['attribute'](attr_output, target_attr)
            if with_attr:
                # define composite loss function
                loss = loss_alpha*feat_loss+(1-loss_alpha)*attr_loss
            else:
                loss = feat_loss 

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            metrics = {"loss": loss.item()}
            # if i % print_interval == 0:
            #     ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            #     self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_samples += len(targets)
                epoch_loss += loss.item()  # add total loss of batch

            # convert feature vector into action class
            # using cosine 
            feat_numpy = feat_output.cpu().detach()
            action_probs = cosine_similarity(feat_numpy, target_feat_met)
            pred_action = np.argmax(action_probs, axis=1)
            metrics["accuracy"] = accuracy_score(y_true=targets.cpu().detach().numpy(), y_pred=pred_action)
            tepoch.set_postfix(metrics)
            
    epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
    return metrics

def train_step1(model, dataloader, optimizer, loss_module, device, class_names, phase='train', l2_reg=False):
    model = model.train()

    epoch_loss = 0  # total loss of epoch
    total_samples = 0  # total samples in epoch

    with tqdm(dataloader, unit="batch", desc=phase) as tepoch:
        for batch in tepoch:
            X, targets, target_feat, target_attr, padding_masks = batch
            # print(X, targets, target_feat, target_attr)
            X = X.float().to(device)
            target_feat = target_feat.float().to(device)
            target_attr = target_attr.float().to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)  # 0s: ignore

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            # with autocast():
                class_output, feat_output = model(X)
                # print("feature shape", feat_output.shape, "target feature shape", target_feat.shape)
                feat_loss = loss_module['feature'](class_output, targets)
                # if with_attr:
                #     attr_loss = loss_module['attribute'](attr_output, target_attr)
   
            loss = feat_loss 

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

            metrics = {"loss": loss.item()}
            # if i % print_interval == 0:
            #     ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            #     self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_samples += len(targets)
                epoch_loss += loss.item()  # add total loss of batch

            # convert feature vector into action class
            # using cosine 
            pred_class = np.argmax(class_output.cpu().detach().numpy(), axis=1)
            metrics["accuracy"] = accuracy_score(y_true=targets.cpu().detach().numpy(), y_pred=pred_class)
            tepoch.set_postfix(metrics)
            
    epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
    return metrics

def eval_step(model, dataloader, loss_module, loss_alpha, device, class_names, target_feat_met, phase='eval', l2_reg=False, with_attrs=True, print_report=True):
    model = model.train()

    epoch_loss = 0  # total loss of epoch
    total_samples = 0  # total samples in epoch
    per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}

    with tqdm(dataloader, unit="batch", desc=phase) as tepoch:
        for batch in tepoch:
            X, targets, target_feat, target_attr, padding_masks = batch
            X = X.float().to(device)
            target_feat = target_feat.float().to(device)
            target_attr = target_attr.float().to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)  # 0s: ignore
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
            # with autocast():
                if with_attrs:
                    attr_output, feat_output = model(X)
                else:
                    feat_output = model(X)
                # print("feature shape", feat_output.shape, "target feature shape", target_feat.shape)
                feat_loss = loss_module['feature'](feat_output, target_feat)
                if with_attrs:
                    attr_loss = loss_module['attribute'](attr_output, target_attr)
            if with_attrs:
                # define composite loss function
                loss = loss_alpha*feat_loss+(1-loss_alpha)*attr_loss
            else:
                loss = feat_loss

            # convert feature vector into action class
            # using cosine 
            feat_numpy = feat_output.cpu().detach()
            action_probs = cosine_similarity(feat_numpy, target_feat_met)
            pred_action = np.argmax(action_probs, axis=1)
            
            per_batch['targets'].append(targets.cpu().numpy())
            per_batch['predictions'].append(pred_action)
            per_batch['metrics'].append([loss.cpu().numpy()])

            tepoch.set_postfix({"loss": loss.item()})
    
    all_preds = np.concatenate(per_batch["predictions"])
    all_targets = np.concatenate(per_batch["targets"])
    metrics_dict = action_evaluator(y_pred=all_preds, y_true=all_targets[:, 0], class_names=class_names, print_report=print_report)
    return metrics_dict

def eval_step1(model, dataloader, loss_module, device, class_names, target_feat_met, phase='seen', l2_reg=False, print_report=True):
    model = model.train()

    epoch_loss = 0  # total loss of epoch
    total_samples = 0  # total samples in epoch
    per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}

    with tqdm(dataloader, unit="batch", desc=phase) as tepoch:
        for batch in tepoch:
            X, targets, target_feat, target_attr, padding_masks = batch
            X = X.float().to(device)
            target_feat = target_feat.float().to(device)
            target_attr = target_attr.float().to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)  # 0s: ignore
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
            # with autocast():

                class_output, feat_output = model(X)
                # print("feature shape", feat_output.shape, "target feature shape", target_feat.shape)
                feat_loss = loss_module['feature'](class_output, targets)
                # if with_attrs:
                #     attr_loss = loss_module['attribute'](attr_output, target_attr)
            # define composite loss function
            loss = feat_loss

            # convert feature vector into action class
            # using cosine 
            if phase == 'seen':
                pred_action = np.argmax(class_output.cpu().detach().numpy(), axis=1)
            else:
                feat_numpy = feat_output.cpu().detach()
                action_probs = cosine_similarity(feat_numpy, target_feat_met)
                pred_action = np.argmax(action_probs, axis=1)
                
            per_batch['targets'].append(targets.cpu().numpy())
            per_batch['predictions'].append(pred_action)
            per_batch['metrics'].append([loss.cpu().numpy()])

            tepoch.set_postfix({"loss": loss.item()})
    
    all_preds = np.concatenate(per_batch["predictions"])
    all_targets = np.concatenate(per_batch["targets"])
    metrics_dict = action_evaluator(y_pred=all_preds, y_true=all_targets[:, 0], class_names=class_names, print_report=print_report)
    return metrics_dict