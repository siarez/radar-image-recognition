import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsample import transforms
from tqdm import tqdm
from models import Net
from utils import ScalarEncoder, accuracy, AverageMeter, make_dataset, save_model, print_metrics
from logger import Logger
from sklearn.model_selection import KFold


data = pd.read_json("data/train.json")
data["band_1"] = data["band_1"].apply(lambda x: np.array(x).reshape(75, 75))
data["band_2"] = data["band_2"].apply(lambda x: np.array(x).reshape(75, 75))
data["inc_angle"] = pd.to_numeric(data["inc_angle"], errors="coerce")


# Augmentation
affine_transforms = transforms.RandomAffine(rotation_range=None, translation_range=0.1, zoom_range=(0.95, 1.05))
rand_flip = transforms.RandomFlip(h=True, v=False)
std_normalize = transforms.StdNormalize()
my_transforms = transforms.Compose([rand_flip, std_normalize])
# scalar encoder for incident angles
encoder = ScalarEncoder(100, 30, 45)
# using folding to create 5 train-validation sets to train 5 networks
kf = KFold(n_splits=5, shuffle=True, random_state=100)
kfold_datasets = []
networks = []
optimizers = []
for train_index, val_index in kf.split(data):
    train_dataset = make_dataset(data.iloc[train_index], encoder, my_transforms)
    val_dataset = make_dataset(data.iloc[val_index], encoder, my_transforms)
    kfold_datasets.append({"train": train_dataset, "val": val_dataset})
    # A new net for each train-validation dataset
    networks.append(Net().cuda())
    optimizers.append(Adam(networks[-1].parameters(), lr=0.0005, weight_decay=0.0002))

# Train
criterion = torch.nn.BCEWithLogitsLoss()
val_criterion = torch.nn.BCELoss()
logger = Logger("./logs")
logger.text_log((str(networks), str(optimizers), str(criterion)), "model_description.txt")


def fit(train, val, batch_size, net, optimizer):
    """
    Runs one epoch on the `net` using the `optimizer`
    :param train: training dataset
    :param val: validation dataset
    :param batch_size: batch size
    :param net: the model to train
    :param optimizer: the optimizer to use on the model
    :return: accuracy and loss performance metrics for training and validation 
    """
    print("train on {} images validate on {} images".format(len(train), len(val)))
    net.train()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    # creating AverageMeters to keep track of the metrics
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
        pbar.set_description("[ loss: {:.4f} | acc: {:.4f} ] ".format(running_loss.avg, running_accuracy.avg))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for val_data_target in val_loader:
        # `trials` is the number of times one sample is run through the model.
        # This combined with dropout gives an idea about confidence of the model about its prediction.
        trials = 10
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
        # prob_out_adjusted = torch.min(torch.max(prob_out_adjusted, Variable(torch.cuda.FloatTensor([0.01]))),
        #                              Variable(torch.cuda.FloatTensor([0.99])))
        # Recording loss and accuracy metrics for `logger`
        val_loss = val_criterion(prob_out_mean, val_target_var)
        val_loss_adjusted = val_criterion(prob_out_adjusted, val_target_var)
        val_acc = accuracy(val_target_var.data, prob_out_mean.data)
        val_acc_mean = accuracy(val_target_var.data, prob_out_adjusted.data)
        val_loss_meter.update(val_loss.data[0])
        val_loss_adjusted_meter.update(val_loss_adjusted.data[0])
        val_acc_meter.update(val_acc)
        val_acc_mean_meter.update(val_acc_mean)
    return [running_loss.avg, running_accuracy.avg, val_loss_meter.avg, val_loss_adjusted_meter.avg,\
               val_acc_meter.avg, val_acc_mean_meter.avg]

prev_loss = 10
patience = 20
for epoch in tqdm(range(150)):
    metrics = []
    for i in range(len(networks)):
        metrics.append(fit(kfold_datasets[i]["train"], kfold_datasets[i]["val"], 32, networks[i], optimizers[i]))
    metrics_avg = np.mean(np.array(metrics), 0)
    print_metrics(epoch+1, metrics_avg)
    # update patience for early stopping
    if prev_loss > metrics_avg[3]:
        prev_loss = metrics_avg[3]
        patience = min(patience + 1, 20)
    else:
        patience -= 1
    print("Patience: ", patience)
    # creating summaries for TensorBoard
    logger.scalar_summary("loss", metrics_avg[0], epoch + 1)
    logger.scalar_summary("vloss", metrics_avg[2], epoch + 1)
    logger.scalar_summary("accuracy", metrics_avg[1], epoch + 1)
    logger.scalar_summary("vloss-adjusted", metrics_avg[3], epoch + 1)
    logger.scalar_summary("v-accuracy", metrics_avg[4], epoch + 1)
    logger.scalar_summary("v-accuracy-adjusted", metrics_avg[5], epoch + 1)
    print("Epoch: ",  epoch + 1)
    if patience == 0:
        # Early stopping
        print("Saving model and Eearly stopping at epoch:", epoch)
        break
save_model(networks, logger.dir)
