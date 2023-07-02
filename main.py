import argparse
import datetime

# params 
parser = argparse.ArgumentParser()
# settings
parser.add_argument('--IMU_data_path', type=str)
parser.add_argument('--I3D_data_path', type=str)
parser.add_argument('--dataset', type=str, default='pamap2', choices=['pamap2', 'daliac', 'mhealth', 'utd-mhad'])
# training
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--imu_alpha', type=float, default=0.0001)
parser.add_argument('--n_epochs', type=float, default=20)
parser.add_argument('--batch_size', type=float, default=64)
# model configuration
parser.add_argument('--d_model', type=float, default=128)
parser.add_argument('--num_heads', type=float, default=2)
parser.add_argument('--feat_size', type=float, default=400)
# data prep params
parser.add_argument('--window_size', type=float, default=5.21)
parser.add_argument('--overlap', type=float, default=4.21)
parser.add_argument('--seq_len', type=float, default=20)
parser.add_argument('--seen_split', type=float, default=0.1)
parser.add_argument('--unseen_split', type=float, default=0.8)

args = parser.parse_args()

# =================================================================

import os 
from datetime import date, datetime
from tqdm.autonotebook import tqdm
from copy import deepcopy
from collections import defaultdict
import numpy as np 
import numpy.random as random
import pandas as pd
import json
import pickle
from collections import defaultdict, OrderedDict

import torch 
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import MSELoss


from src.datasets.data import PAMAP2ReaderV2, DaLiAcReaderV2, MHEALTHReaderV2, UTDReader
# from src.datasets.dataset import PAMAP2Dataset
from src.utils.analysis import action_evaluator
from src.datasets.utils import load_attribute

from src.models.loss import FeatureLoss, AttributeLoss
from src.utils.analysis import action_evaluator

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
# from umap import UMAP

import matplotlib.pyplot as plt 
import seaborn as sns 

args = {}

# setup env data 
if args.device == 'cpu':
    device = "cpu"
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fold_mapping = {
   'pamap2': [['watching TV', 'house cleaning', 'standing', 'ascending stairs'], ['walking', 'rope jumping', 'sitting', 'descending stairs'], ['playing soccer', 'lying', 'vacuum cleaning', 'computer work'], ['cycling', 'running', 'Nordic walking'], ['ironing', 'car driving', 'folding laundry']],
   'daliac': [['sitting', 'vacuuming', 'descending stairs'], ['lying', 'sweeping', 'treadmill running'], ['standing', 'walking', 'cycling'], ['washing dishes', 'ascending stairs', 'rope jumping']],
   'mhealth': [['Standing still', 'Climbing stairs', 'Cycling'], ['Sitting and relaxing', 'Waist bends forward', 'Jogging'], ['Lying down', 'Frontal elevation of arms', 'Running'], ['Walking', 'Knees bending (crouching)', 'Jump front & back']],
   'utd-mhad': [['swipe left', 'arm cross', 'draw triangle', 'arm curl', 'jog', 'pickup & throw'], ['swipe right', 'basketball shoot', 'bowling', 'tennis serve', 'walk', 'squat'], ['wave', 'draw x', 'boxing', 'push', 'sit to stand'], ['clap', 'draw circle(clockwise)', 'baseball swing', 'knock', 'stand to sit'], ['throw', 'draw circle(counter clockwise)', 'tennis swing', 'catch', 'lunge']]
}
fold_classes = fold_mapping[args.dataset]
fold_cls_ids = [[actionList.index(i) for i in j] for j in fold_classes]

def save_model(model, save_path, unique_name, fold_id):
    os.makedirs(save_path,exist_ok=True)
    torch.save({
        "n_epochs" : args.n_epochs,
        "model_state_dict":model.state_dict(),
        "config": config
    }, f"{save_path}/{unique_name}_{fold_id}.pt")

# load imu data 
loader_mapping = {
    'pamap2': PAMAP2ReaderV2,
    'daliac': DaLiAcReaderV2,
    'mhealth': MHEALTHReaderV2,
    'utd-mhad': UTDReader
}
dataReader = loader_mapping[args.dataset](args.IMU_data_path)
actionList = dataReader.idToLabel

# load auxiliary data
def read_I3D_pkl(loc,feat_size="400"):
  if feat_size == "400":
    feat_index = 1
  elif feat_size == "2048":
    feat_index = 0
  else:
    raise NotImplementedError()

  with open(loc,"rb") as f0:
    __data = pickle.load(f0)

  label = []
  prototype = []
  for k,v in __data.items():
    label.append(k)
    all_arr = [x[feat_index] for x in v]
    all_arr = np.asarray(all_arr).mean(axis=0)
    prototype.append(all_arr)

  label = np.asarray(label)
  prototype = np.array(prototype)
  return {"activity":label, "features":prototype}

video_data = read_I3D_pkl(args.I3D_data_path,feat_size="400")
video_classes, video_feat = video_data['activity'], video_data['features']

# load I3D data 
def selecting_video_prototypes(prototypes:np.array,classes:np.array,vid_class_name:np.array):
    selected = []
    for tar in vid_class_name:
        indexes = np.where(classes == tar)
        selected.append(torch.from_numpy(prototypes[random.choice(indexes[0])]))

    return torch.stack(selected)

label2Id = {c[1]:i for i,c in enumerate(dataReader.label_map)}
action_dict = defaultdict(list)
skeleton_Ids = []
for i, a in enumerate(video_classes):
    action_dict[label2Id[a]].append(i)
    skeleton_Ids.append(label2Id[a])

# Dataset Class definition 
class PAMAP2Dataset(Dataset):
    def __init__(self, data, actions, attributes, attribute_dict, action_classes, seq_len=120):
        super(PAMAP2Dataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.actions = actions
        self.attribute_dict = attribute_dict
        self.seq_len = seq_len
        self.attributes = torch.from_numpy(attributes)
        self.action_classes = action_classes
        # build action to id mapping dict
        self.n_action = len(self.actions)
        self.action2Id = dict(zip(action_classes, range(self.n_action)))

    def __getitem__(self, ind):
        x = self.data[ind, ...]
        target = self.actions[ind]
        y = torch.from_numpy(np.array([self.action2Id[target]]))
        # extraction semantic space generation skeleton sequences
        vid_idx = random.choice(self.attribute_dict[target])
        y_feat = self.attributes[vid_idx, ...]
        return x, y, y_feat

    def __len__(self):
        return self.data.shape[0]

    def getClassAttrs(self):
        sampling_idx = [random.choice(self.attribute_dict[i]) for i in self.action_classes]
        ft_mat = self.attributes[sampling_idx, ...]
        return ft_mat

    def getClassFeatures(self):
        cls_feat = []
        for c in self.action_classes:
            idx = self.attribute_dict[c]
            cls_feat.append(torch.mean(self.attributes[idx, ...], dim=0))

        cls_feat = torch.vstack(cls_feat)
        # print(cls_feat.size())
        return cls_feat
    
class IMUEncoder(nn.Module):
    def __init__(self, in_ft, d_model, ft_size, n_classes, num_heads=1, max_len=1024, dropout=0.1):
        super(IMUEncoder, self).__init__()
        self.in_ft = in_ft
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.ft_size = ft_size 
        self.n_classes = n_classes

        self.lstm = nn.LSTM(input_size=self.in_ft,
                            hidden_size=self.d_model,
                            num_layers=self.num_heads,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)
        self.drop = nn.Dropout(p=0.1)
        self.act = nn.ReLU()
        self.fcLayer1 = nn.Linear(2*self.d_model, self.ft_size)
        # self.fcLayer2 = nn.Linear(self.ft_size, self.ft_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out_forward = out[:, self.max_len - 1, :self.d_model]
        out_reverse = out[:, 0, self.d_model:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        out = self.drop(out_reduced)
        out = self.act(out)
        out = self.fcLayer1(out)
        # out = self.fcLayer2(out)
        return out
    
# setup objective functions & steps

def loss_cross_entropy(y_pred, y, feat, loss_fn):
    mm_vec = torch.mm(y_pred, torch.transpose(feat, 0, 1))
    feat_norm = torch.norm(feat, p=2, dim=1)
    norm_vec = mm_vec/torch.unsqueeze(feat_norm, 0)
    softmax_vec = torch.softmax(norm_vec, dim=1)
    output = loss_fn(softmax_vec, y)
    pred = torch.argmax(softmax_vec, dim=-1)
    return output, pred

def shuffledTripletLoss(pred_feat, sem_space, y, bs=32, loss_fn=nn.TripletMarginLoss(margin=0.1, p=2, reduction='none')):
    anchor_feat = sem_space[y, ...]
    neg_feat = torch.concat([anchor_feat[bs//2:, ...], anchor_feat[:bs//2, ...]], dim=0).squeeze()
    pos_feat = anchor_feat.squeeze()

    neg_y = torch.concat([y[bs//2:], y[:bs//2]])
    y_mask = (y!=neg_y).long()

    output_arr = loss_fn(pred_feat, pos_feat, neg_feat)
    masked_arr = torch.multiply(output_arr, y_mask)

    output = masked_arr.mean()
    return output

def loss_reconstruction_calc(y_pred, y_feat, loss_fn=nn.L1Loss(reduction="sum")):
    loss = loss_fn(y_pred,y_feat)
    return loss

def predict_class(y_pred, feat):
    mm_vec = torch.mm(y_pred, torch.transpose(feat, 0, 1))
    feat_norm = torch.norm(feat, p=2, dim=1)
    norm_vec = mm_vec/torch.unsqueeze(feat_norm, 0)
    softmax_vec = torch.softmax(norm_vec, dim=1)
    pred = torch.argmax(softmax_vec, dim=-1)
    return pred

def train_step(model, dataloader, dataset:PAMAP2Dataset, optimizer, loss_module, device, class_names, phase='train', l2_reg=False, loss_alpha=0.7):
    model = model.train()
    epoch_loss = 0  # total loss of epoch
    total_samples = 0  # total samples in epoch
    random_selected_feat = dataset.getClassFeatures().to(device)

    with tqdm(dataloader, unit="batch", desc=phase) as tepoch:
        for batch in tepoch:
            X, targets, target_feat = batch
            X = X.float().to(device)
            target_feat = target_feat.float().to(device)
            targets = targets.long().to(device)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            # with autocast():
                feat_output = model(X)
                class_loss, class_output = loss_cross_entropy(feat_output, targets.squeeze(), random_selected_feat, loss_fn =loss_module['class'] )
                const_loss = shuffledTripletLoss(feat_output, random_selected_feat, targets, loss_fn=loss_module["constrastive"])
                feat_loss = loss_reconstruction_calc(feat_output,target_feat,loss_fn=loss_module["feature"])

            #loss = cross_entropy_loss
            loss = feat_loss + loss_alpha*class_loss + loss_alpha*const_loss
            # class_output = predict_class(feat_output,random_selected_feat)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            metrics = {"loss": loss.item()}
            with torch.no_grad():
                total_samples += len(targets)
                epoch_loss += loss.item()  # add total loss of batch

            # convert feature vector into action class
            # using cosine
            pred_class = class_output.cpu().detach().numpy()
            metrics["accuracy"] = accuracy_score(y_true=targets.cpu().detach().numpy(), y_pred=pred_class)
            tepoch.set_postfix(metrics)

    epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
    return metrics

def eval_step(model, dataloader,dataset, loss_module, device, class_names,  phase='seen', l2_reg=False, print_report=False, show_plot=False, loss_alpha=0.7):
    model = model.eval()
    random_selected_feat = dataset.getClassFeatures().to(device)
    epoch_loss = 0  # total loss of epoch
    total_samples = 0  # total samples in epoch
    per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
    metrics = {"samples": 0, "loss": 0, "feat. loss": 0, "classi. loss": 0}

    with tqdm(dataloader, unit="batch", desc=phase) as tepoch:
        for batch in tepoch:
            X, targets, target_feat = batch
            X = X.float().to(device)
            X = X.float().to(device)
            target_feat = target_feat.float().to(device)
            targets = targets.long().to(device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            # with autocast():
                feat_output = model(X)
                class_loss, class_output = loss_cross_entropy(feat_output,targets.squeeze(),random_selected_feat,loss_fn =loss_module['class'] )
                const_loss = shuffledTripletLoss(feat_output, random_selected_feat, targets, loss_fn=loss_module["constrastive"])
                feat_loss = loss_reconstruction_calc(feat_output,target_feat,loss_fn=loss_module["feature"])
            
            #loss = cross_entropy_loss
            loss = feat_loss + loss_alpha*class_loss + loss_alpha*const_loss
            # class_output = predict_class(feat_output,random_selected_feat)

            pred_action = class_output

            with torch.no_grad():
                metrics['samples'] += len(targets)
                metrics['loss'] += loss.item()  # add total loss of batch
                metrics['feat. loss'] += feat_loss.item()
                metrics['classi. loss'] += class_loss.item()

            per_batch['targets'].append(targets.cpu().numpy())
            per_batch['predictions'].append(pred_action.cpu().numpy())
            per_batch['metrics'].append([loss.cpu().numpy()])

            tepoch.set_postfix({"loss": loss.item()})

    all_preds = np.concatenate(per_batch["predictions"])
    all_targets = np.concatenate(per_batch["targets"])
    metrics_dict = action_evaluator(y_pred=all_preds, y_true=all_targets[:, 0], class_names=class_names, print_report=print_report, show_plot=show_plot)
    metrics_dict.update(metrics)
    return metrics_dict

# setup log functions 

def plot_curves(df):
    df['loss'] = df['loss']/df['samples']
    df['feat. loss'] = df['feat. loss']/df['samples']
    df['classi. loss'] = df['classi. loss']/df['samples']
    
    fig, axs = plt.subplots(nrows=4)
    sns.lineplot(data=df, x='epoch', y='loss', hue='phase', marker='o', ax=axs[2]).set(title="Loss")
    sns.lineplot(data=df, x='epoch', y='feat. loss', hue='phase', marker='o', ax=axs[0]).set(title="Feature Loss")
    sns.lineplot(data=df, x='epoch', y='classi. loss', hue='phase', marker='o', ax=axs[1]).set(title="Classification Loss")
    sns.lineplot(data=df, x='epoch', y='accuracy', hue='phase', marker='o', ax=axs[3]).set(title="Accuracy")


if __name__ == 'main':
    fold_metric_scores = []

    for i, cs in enumerate(fold_cls_ids):
        print("="*16, f'Fold-{i}', "="*16)
        print(f'Unseen Classes : {fold_classes[i]}')

        data_dict = dataReader.generate(unseen_classes=cs, seen_ratio=args.seen_split, unseen_ratio=args.unseen_split, window_size=args.window_size, window_overlap=args.overlap, resample_freq=args.seq_len)
        all_classes = dataReader.idToLabel
        seen_classes = data_dict['seen_classes']
        unseen_classes = data_dict['unseen_classes']
        print("seen classes > ", seen_classes)
        print("unseen classes > ", unseen_classes)
        train_n, seq_len, in_ft = data_dict['train']['X'].shape

        print("Initiate IMU datasets ...")
        # build IMU datasets
        train_dt = PAMAP2Dataset(data=data_dict['train']['X'], actions=data_dict['train']['y'], attributes=video_feat, attribute_dict=action_dict, action_classes=seen_classes, seq_len=100)
        train_dl = DataLoader(train_dt, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        # build seen eval_dt
        eval_dt = PAMAP2Dataset(data=data_dict['eval-seen']['X'], actions=data_dict['eval-seen']['y'], attributes=video_feat, attribute_dict=action_dict, action_classes=seen_classes, seq_len=100)
        eval_dl = DataLoader(eval_dt, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        # build unseen test_dt
        test_dt = PAMAP2Dataset(data=data_dict['test']['X'], actions=data_dict['test']['y'], attributes=video_feat, attribute_dict=action_dict, action_classes=unseen_classes, seq_len=100)
        test_dl = DataLoader(test_dt, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        
        # build model
        imu_config = {
            'in_ft':in_ft, 
            'd_model':args.d_model, 
            'num_heads':args.num_heads, 
            'ft_size':args.feat_size, 
            'max_len':seq_len, 
            'n_classes':len(seen_classes)
        }
        model = IMUEncoder(**imu_config)
        model.to(device)

        # define run parameters 
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
        loss_module = {'class': nn.CrossEntropyLoss(reduction="sum"), 'constrastive': nn.TripletMarginLoss(margin=0.5, p=1, reduction='none'), 'feature': nn.L1Loss(reduction="sum")}
        best_acc = 0.0

        # train the model 
        train_data = []
        for epoch in tqdm(range(args.n_epochs), desc='Training Epoch', leave=False):
        
            train_metrics = train_step(model, train_dl, train_dt,optimizer, loss_module, device, class_names=[all_classes[i] for i in seen_classes], phase='train', loss_alpha=0.0001)
            train_metrics['epoch'] = epoch
            train_metrics['phase'] = 'train'
            train_data.append(train_metrics)

            eval_metrics = eval_step(model, eval_dl, eval_dt,loss_module, device, class_names=[all_classes[i] for i in seen_classes], phase='seen', loss_alpha=0.0001, print_report=False, show_plot=False)
            eval_metrics['epoch'] = epoch 
            eval_metrics['phase'] = 'valid'
            train_data.append(eval_metrics)
            # print(f"EPOCH [{epoch}] TRAINING : {train_metrics}")
            # print(f"EPOCH [{epoch}] EVAL : {eval_metrics}")
            if eval_metrics['accuracy'] > best_acc:
                best_model = deepcopy(model.state_dict())
        
        train_df = pd.DataFrame().from_records(train_data)
        plot_curves(train_df)

        # replace by best model 
        model.load_state_dict(best_model)
        save_model(model, args.save_path, datetime.datetime.now().strftime("%Y.%m.%d-%H%M%S"), i)

        # run evaluation on unseen classes
        test_metrics = eval_step(model, test_dl,test_dt, loss_module, device, class_names=[all_classes[i] for i in unseen_classes], phase='unseen', loss_alpha=0.0001, print_report=True, show_plot=True)
        test_metrics['N'] = len(unseen_classes)
        fold_metric_scores.append(test_metrics)
        print(test_metrics)
        print("="*40)

    print("="*14, "Overall Unseen Classes Performance", "="*14)
    seen_score_df = pd.DataFrame.from_records(fold_metric_scores)
    weighted_score_df = seen_score_df[["accuracy", "precision", "recall", "f1"]].multiply(seen_score_df["N"], axis="index")
    final_results = weighted_score_df.sum()/seen_score_df['N'].sum()
    print(final_results)

