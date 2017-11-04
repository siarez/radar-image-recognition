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
from logger import Logger

run_infer = True

data = pd.read_json('train.json')

if run_infer:
    test = pd.read_json('test.json')
    test['band_1'] = test['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    test['band_2'] = test['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
    band_1_test = np.concatenate([im for im in test['band_1']]).reshape(-1, 75, 75)
    band_2_test = np.concatenate([im for im in test['band_2']]).reshape(-1, 75, 75)
    inc_angle_test = np.nan_to_num(test['inc_angle'].values)
    inc_angle_test = np.zeros(test['inc_angle'].values.shape)
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
inc_angle_tr = np.nan_to_num(train['inc_angle'].values)
inc_angle_tr = np.zeros(train['inc_angle'].values.shape)
full_img_tr = np.stack([band_1_tr, band_2_tr], axis=1)

band_1_val = np.concatenate([im for im in val['band_1']]).reshape(-1, 75, 75)
band_2_val = np.concatenate([im for im in val['band_2']]).reshape(-1, 75, 75)
inc_angle_val = np.nan_to_num(val['inc_angle'].values)
inc_angle_val = np.zeros(val['inc_angle'].values.shape)
full_img_val = np.stack([band_1_val, band_2_val], axis=1)

del data

# Augmentation
affine_transforms = transforms.RandomAffine(rotation_range=None, translation_range=0.2, zoom_range=(0.8, 1.2))
rand_flip = transforms.RandomFlip(h=True, v=False)

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
val_criterion = nn.BCELoss()
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
    #y_true = y_true.float()
    #_, y_pred = torch.max(y_pred, dim=-1)
    y_pred_decision = y_pred > 0.5
    return (y_pred_decision.float() == y_true.float()).float().mean()


def fit(train, val, epochs, batch_size):
    # Set the logger
    logger = Logger('./logs')
    print('train on {} images validate on {} images'.format(len(train), len(val)))
    net.train()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    for epoch in tqdm(range(epochs), total=epochs):
        running_loss = AverageMeter()
        running_accuracy = AverageMeter()
        val_loss_meter = AverageMeter()
        val_loss_adjusted_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        val_acc_mean_meter = AverageMeter()
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
        print("\n[ loss: {:.4f} | acc: {:.4f} ] ".format(running_loss.avg, running_accuracy.avg))
        for val_data_target in val_loader:
            trials = 10  # number of times one sample is run through the model. This combined with dropout give you an idea about confidence of the model about its prediction.
            current_batch_size = val_data_target[0][0].size()[0]  # Because last batch can have a different size
            val_img_var, val_target_var = Variable(val_data_target[0][0].float().cuda()), Variable(val_data_target[0][1].float().cuda())
            val_img_var = torch.stack([val_img_var] * trials, 1).view(current_batch_size*trials, 2, 75, 75)
            val_angle_var = Variable(val_data_target[1].float().cuda())
            val_angle_var = torch.stack([val_angle_var] * trials, 1).view(current_batch_size*trials, 1)
            output_logits = torch.squeeze(net(val_img_var, val_angle_var))
            prob_out = torch.sigmoid(output_logits)
            prob_out = prob_out.view(current_batch_size, trials)
            prob_out_mean = torch.mean(prob_out, 1)
            prob_out_std = torch.std(prob_out, 1)
            prob_out_adjusted = (1 - 2 * prob_out_std)*(prob_out_mean - 0.5) + 0.5
            prob_out_adjusted = torch.min(torch.max(prob_out_adjusted, Variable(torch.cuda.FloatTensor([0.05]))),
                                          Variable(torch.cuda.FloatTensor([0.95])))
            val_loss = val_criterion(prob_out[:, 0], val_target_var)
            val_loss_adjusted = val_criterion(prob_out_adjusted, val_target_var)
            val_acc = accuracy(val_target_var.data, prob_out[:, 0].data)
            val_acc_mean = accuracy(val_target_var.data, prob_out_mean.data)
            val_loss_meter.update(val_loss.data[0])
            val_loss_adjusted_meter.update(val_loss_adjusted.data[0])
            val_acc_meter.update(val_acc)
            val_acc_mean_meter.update(val_acc_mean)
        print("\n[Epoch: {:} loss: {:.4f} | acc: {:.4f} | vloss: {:.4f} | vadjloss: {:.4f} | vacc: {:.4f} | vacc_m: {:.4f} ] ".format(
            epoch, running_loss.avg, running_accuracy.avg, val_loss_meter.avg, val_loss_adjusted_meter.avg, val_acc_meter.avg, val_acc_mean_meter.avg))
        logger.scalar_summary("loss", running_loss.avg, epoch + 1)
        logger.scalar_summary("vloss", val_loss_meter.avg, epoch + 1)
        logger.scalar_summary("vloss-adjusted", val_loss_adjusted_meter.avg, epoch + 1)
        logger.scalar_summary("accuracy", running_accuracy.avg, epoch + 1)
        logger.scalar_summary("v-accuracy", val_acc_meter.avg, epoch + 1)
        logger.scalar_summary("v-accuracy-adjusted", val_acc_mean_meter.avg, epoch + 1)

fit(train_dataset, val_dataset, 250, 32)

if run_infer:
    print('Running test data')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_pred_mean = torch.FloatTensor()
    test_pred_std = torch.FloatTensor()
    for data_target in tqdm(test_loader, total=len(test_loader)):
        data_var_img = Variable(data_target[0][0].float().cuda())
        data_var_angle = Variable(data_target[1].float().cuda())
        _data_dout_img = data_var_img.repeat(10, 1, 1, 1)
        _data_dout_angel = data_var_angle.repeat(10, 1)
        out_logits = net(_data_dout_img, _data_dout_angel).data
        out_guesses = torch.sigmoid(out_logits)
        out_mean = torch.mean(out_guesses)
        out_std = torch.std(out_guesses)
        test_pred_mean = torch.cat((test_pred_mean, torch.FloatTensor([out_mean])), 0)
        test_pred_std = torch.cat((test_pred_std, torch.FloatTensor([out_std])), 0)

    test_pred_std.squeeze_()
    print(test_pred_std[0:20])
    test_pred_mean.squeeze_()
    prob_out_adjusted = torch.min(torch.max(test_pred_mean, torch.FloatTensor([0.05])), torch.FloatTensor([0.95]))
    prob_out_adjusted = (1 - 2 * test_pred_std) * (prob_out_adjusted - 0.5) + 0.5
    pred_probability = prob_out_adjusted.cpu().numpy()
    pred_df = pd.DataFrame(index=test.index)
    pred_df['id'] = test['id']
    pred_df['is_iceberg'] = pd.Series(pred_probability, index=test.index).astype(float)
    pred_df['is_iceberg'] = pred_df['is_iceberg'].apply("{0:.5f}".format)


    pred_df.to_csv('submission_prob.csv', encoding='utf-8', columns=['id', 'is_iceberg'], index=False)
