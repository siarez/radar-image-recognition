import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as tv_trans
from torchsample import transforms, TensorDataset
from tqdm import tqdm
from models import Net, CNNClassifier
from multi_dataset import MultiDataset


run_infer = True

data = pd.read_json('train.json')

if run_infer:
    test = pd.read_json('train.json')
    test['band_1'] = test['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    test['band_2'] = test['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
    band_1_test = np.concatenate([im for im in test['band_1']]).reshape(-1, 75, 75)
    band_2_test = np.concatenate([im for im in test['band_2']]).reshape(-1, 75, 75)
    inc_angle_test = test['inc_angle'].values
    full_img_test = np.stack([band_1_test, band_2_test], axis=1)
    test_imgs = torch.from_numpy(full_img_test).float().cuda()
    test_angles = torch.from_numpy(inc_angle_test).float().unsqueeze_(1)
    test_dataset_img = TensorDataset(test_imgs, torch.zeros(test_imgs.shape[0]))
    test_dataset_angles = TensorDataset(test_angles, None)
    test_dataset = MultiDataset((test_dataset_img, test_dataset_angles))


data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

train = data.sample(frac=0.8)
val = data[~data.isin(train)].dropna()

# Concat Bands into (N, 2, 75, 75) images
band_1_tr = np.concatenate([im for im in train['band_1']]).reshape(-1, 75, 75)
band_2_tr = np.concatenate([im for im in train['band_2']]).reshape(-1, 75, 75)
inc_angle_tr = train['inc_angle'].values
full_img_tr = np.stack([band_1_tr, band_2_tr], axis=1)

band_1_val = np.concatenate([im for im in val['band_1']]).reshape(-1, 75, 75)
band_2_val = np.concatenate([im for im in val['band_2']]).reshape(-1, 75, 75)
inc_angle_val = val['inc_angle'].values
full_img_val = np.stack([band_1_val, band_2_val], axis=1)

del data

# Augmentation
affine_transforms = transforms.RandomAffine(rotation_range=180, translation_range=0.2, zoom_range=(0.8, 1.2))
rand_flip = transforms.RandomFlip(h=True, v=True)
my_transforms = transforms.Compose([affine_transforms, rand_flip])

# Dataset and DataLoader
train_imgs = torch.from_numpy(full_img_tr).float()
train_angles = torch.from_numpy(inc_angle_tr).float().unsqueeze_(1)
train_targets = torch.from_numpy(train['is_iceberg'].values).long()
train_dataset_imgs = TensorDataset(train_imgs, train_targets, input_transform=my_transforms)
train_dataset_angles = TensorDataset(train_angles, None)
train_dataset = MultiDataset((train_dataset_imgs, train_dataset_angles))

val_imgs = torch.from_numpy(full_img_val).float()
val_angles = torch.from_numpy(inc_angle_val).float().unsqueeze_(1)
val_targets = torch.from_numpy(val['is_iceberg'].values).long()
val_dataset_img = TensorDataset(val_imgs, val_targets, input_transform=my_transforms)
val_dataset_angles = TensorDataset(val_angles, None)
val_dataset = MultiDataset((val_dataset_img, val_dataset_angles))

net = Net()

img_size = (75, 75)
img_ch = 2
kernel_size = 3
pool_size = 2
n_out = 1
#net = CNNClassifier(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, pool_size=pool_size, n_out=n_out)

net.cuda()

# Train
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(net.parameters(), lr=0.001)


# utils
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(y_true, y_pred):
    y_true = y_true.float()
    #_, y_pred = torch.max(y_pred, dim=-1)
    y_pred = (torch.sigmoid(y_pred) > 0.5)
    return (y_pred.float() == y_true).float().mean()


def fit(train, val, epochs, batch_size):
    print('train on {} images validate on {} images'.format(len(train), len(val)))
    net.train()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    for epoch in tqdm(range(epochs), total=epochs):
        running_loss = AverageMeter()
        running_accuracy = AverageMeter()
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        pbar = tqdm(train_loader, total=len(train_loader))
        for data_target in pbar:
            data_var_img, target_var = Variable(data_target[0][0].float().cuda()), Variable(data_target[0][1].float().cuda())
            data_var_angle = Variable(data_target[1].float().cuda())
            output = torch.squeeze(net(data_var_img, data_var_angle))
            loss = criterion(output, target_var)
            acc = accuracy(target_var.data, output.data)
            running_loss.update(loss.data[0])
            running_accuracy.update(acc)
            pbar.set_description("[ loss: {:.4f} | acc: {:.4f} ] ".format(
                running_loss.avg, running_accuracy.avg))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print("[ loss: {:.4f} | acc: {:.4f} ] ".format(running_loss.avg, running_accuracy.avg))
        for val_data_target in val_loader:
            val_var_img, val_target_var = Variable(val_data_target[0][0].float().cuda()), Variable(val_data_target[0][1].float().cuda())
            val_var_angle = Variable(val_data_target[1].float().cuda())
            output = torch.squeeze(net(val_var_img, val_var_angle))
            val_loss = criterion(output, val_target_var)
            val_acc = accuracy(val_target_var.data, output.data)
            val_loss_meter.update(val_loss.data[0])
            val_acc_meter.update(val_acc)
        pbar.set_description("[ loss: {:.4f} | acc: {:.4f} | vloss: {:.4f} | vacc: {:.4f} ] ".format(
            running_loss.avg, running_accuracy.avg, val_loss_meter.avg, val_acc_meter.avg))
        print("[Epoch: {:} loss: {:.4f} | acc: {:.4f} | vloss: {:.4f} | vacc: {:.4f} ] ".format(
            epoch, running_loss.avg, running_accuracy.avg, val_loss_meter.avg, val_acc_meter.avg))


fit(train_dataset, val_dataset, 10, 32)

if run_infer:
    print('Running test data')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_pred = torch.FloatTensor()
    test_pred_conf = torch.FloatTensor()
    for data_target in tqdm(test_loader, total=len(test_loader)):
        data_var_img = Variable(data_target[0][0].float().cuda())
        data_var_angle = Variable(data_target[1].float().cuda())
        _data_dout_img = data_var_img.repeat(10, 1, 1, 1)
        _data_dout_angel = data_var_angle.repeat(10, 1)
        out_logits = net(_data_dout_img, _data_dout_angel).data
        out_guesses = torch.sigmoid(out_logits)
        out_mean = torch.mean(out_guesses)
        out_std = torch.std(out_guesses)
        test_pred = torch.cat((test_pred, torch.FloatTensor([out_mean])), 0)
        test_pred_conf = torch.cat((test_pred_conf, torch.FloatTensor([out_std])), 0)

    test_pred_conf.squeeze_()
    print(test_pred_conf[0:20])
    test_pred.squeeze_()
    pred_probability = torch.min(torch.max(test_pred, torch.FloatTensor([0.05])), torch.FloatTensor([0.95]))
    pred_probability = pred_probability.cpu().numpy()
    pred_df = pd.DataFrame(index=test.index)
    pred_df['id'] = test['id']
    pred_df['is_iceberg'] = pd.Series(pred_probability, index=test.index).astype(float)
    pred_df['is_iceberg'] = pred_df['is_iceberg'].apply("{0:.5f}".format)


    pred_df.to_csv('submission_prob.csv', encoding='utf-8', columns=['id', 'is_iceberg'], index=False)
