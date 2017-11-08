import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsample import transforms
from tqdm import tqdm
from models import Net
from utils import ScalarEncoder, accuracy, AverageMeter, make_dataset
from logger import Logger
import os.path


run_infer = True

data = pd.read_json('train.json')
data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')

train = data.sample(frac=0.8)
val = data[~data.isin(train)].dropna()

# Augmentation
affine_transforms = transforms.RandomAffine(rotation_range=None, translation_range=0.2, zoom_range=(0.8, 1.2))
rand_flip = transforms.RandomFlip(h=True, v=False)
std_normalize = transforms.StdNormalize()
my_transforms = transforms.Compose([rand_flip, std_normalize])
# scalar encoder for incident angles
encoder = ScalarEncoder(100, 30, 45)

train_dataset = make_dataset(train, encoder, my_transforms)
val_dataset = make_dataset(val, encoder, my_transforms)

net = Net()
# net = CNNClassifier(img_size=(75, 75), img_ch=2, kernel_size=3, pool_size=2, n_out=1)
net.cuda()

# Train
criterion = torch.nn.BCEWithLogitsLoss()
val_criterion = torch.nn.BCELoss()
optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.001)
logger = Logger('./logs')


def fit(train, val, epochs, batch_size):
    # Set the logger
    logger.text_log((str(net), str(optimizer), str(criterion)), "model_description.txt")
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
            # wrapping tensors in autograd variables
            val_img_var = Variable(val_data_target[0][0].float().cuda())
            val_target_var = Variable(val_data_target[0][1].float().cuda())
            val_angle_var = Variable(val_data_target[1].float().cuda())
            # Forward pass
            output_logits = torch.squeeze(net(val_img_var, val_angle_var, trials=trials))
            prob_out = torch.sigmoid(output_logits)
            prob_out = prob_out.view(current_batch_size, trials)
            # Taking mean and std of trails for each sample
            prob_out_mean = torch.mean(prob_out, 1)
            prob_out_std = torch.std(prob_out, 1)
            # Adjusting the mean probability of trials according to their std.
            # As std increases, the ceiling is lowered and floor is raised.
            # std=0 means floor and ceiling are untouched. std=0.5 floor and ceiling are both at 0.5
            prob_out_adjusted = (1 - 2 * prob_out_std)*(prob_out_mean - 0.5) + 0.5
            #prob_out_adjusted = torch.min(torch.max(prob_out_adjusted, Variable(torch.cuda.FloatTensor([0.05]))),
            #                              Variable(torch.cuda.FloatTensor([0.95])))
            # Recording loss and accuracy metrics for `logger`
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
        # creating summaries for TensorBoard
        logger.scalar_summary("loss", running_loss.avg, epoch + 1)
        logger.scalar_summary("vloss", val_loss_meter.avg, epoch + 1)
        logger.scalar_summary("vloss-adjusted", val_loss_adjusted_meter.avg, epoch + 1)
        logger.scalar_summary("accuracy", running_accuracy.avg, epoch + 1)
        logger.scalar_summary("v-accuracy", val_acc_meter.avg, epoch + 1)
        logger.scalar_summary("v-accuracy-adjusted", val_acc_mean_meter.avg, epoch + 1)

fit(train_dataset, val_dataset, 300, 32)

if run_infer:
    print("Loading testset")
    test_df = pd.read_json('test.json')
    test_df['band_1'] = test_df['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    test_df['band_2'] = test_df['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    test_df['inc_angle'] = pd.to_numeric(test_df['inc_angle'], errors='coerce')
    test_dataset = make_dataset(test_df, encoder, my_transforms, test=True)

    print('Running test data')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_pred_mean = torch.FloatTensor()
    test_pred_std = torch.FloatTensor()
    num_to_sample = 10
    for data_target in tqdm(test_loader, total=len(test_loader)):
        data_var_img = Variable(data_target[0][0].float().cuda())
        data_var_angle = Variable(data_target[1].float().cuda())
        out_logits = net(data_var_img, data_var_angle, trials=num_to_sample).data
        out_guesses = torch.sigmoid(out_logits)
        out_mean = torch.mean(out_guesses)
        out_std = torch.std(out_guesses)
        test_pred_mean = torch.cat((test_pred_mean, torch.FloatTensor([out_mean])), 0)
        test_pred_std = torch.cat((test_pred_std, torch.FloatTensor([out_std])), 0)

    test_pred_std.squeeze_()
    print(test_pred_std[0:10])  # to get a peek
    test_pred_mean.squeeze_()
    prob_out_adjusted = (1 - 2 * test_pred_std) * (test_pred_mean - 0.5) + 0.5
    # prob_out_adjusted = torch.min(torch.max(test_pred_mean, torch.FloatTensor([0.05])), torch.FloatTensor([0.95]))
    pred_probability = prob_out_adjusted.cpu().numpy()
    pred_df = pd.DataFrame(index=test_df.index)
    pred_df['id'] = test_df['id']
    pred_df['is_iceberg'] = pd.Series(pred_probability, index=test_df.index).astype(float)
    # removing scientific notation because I'm not sure how Kaggle deasl with it.
    pred_df['is_iceberg'] = pred_df['is_iceberg'].apply("{0:.9f}".format)


    pred_df.to_csv(os.path.join(logger.dir, 'submission_prob.csv'), encoding='utf-8', columns=['id', 'is_iceberg'], index=False)
