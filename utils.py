import numpy as np
from torch.utils.data import dataset
import torch
from torch.autograd import Variable
from torchsample import TensorDataset
import os
from models import Net

class ScalarEncoder:
    """
    A class for one-hot encoding scalars
    """
    def __init__(self, num_of_bins, min, max):
        self.num_bins = num_of_bins  # possible values of the one-hot encoded representation.
        self.min = min
        self.max = max
        self.multiplier = (self.num_bins - 2) / (self.max - self.min)
        self.lookup = np.eye(self.num_bins)

    def encode(self, input_arr):
        """
        One-hot encodes 1D array of scalars . Scalars that fall bellow the `min` will have the same encoding.
        Same is true for the ones that land above `max`
        :param input_arr: 1D array of scalars
        :return: 2D one array of size ( len(`input_arr`) , `num_of_bins`)
        """
        scaled_input = np.array((input_arr - self.min) * self.multiplier).astype(np.int)
        scaled_input = np.minimum(scaled_input, self.num_bins - 1)
        scaled_input = np.maximum(scaled_input, 0)
        encoded = self.lookup[scaled_input, :]
        encoded[:, 0] = 0  # removing n/a inc angles, to prevent leakage
        return encoded


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
    """Given predicted probabilities and targets calculates accuracy."""
    y_pred_decision = y_pred > 0.5
    return (y_pred_decision.float() == y_true.float()).float().mean()


class MultiDataset(dataset.Dataset):
    """
    Built-in datasets only one tensor as input. Sometimes it does not make sense to store all the input in one tensor.
    In this case incident angle and images can't be put in the same tensor.
    This class allows for iterating through different datasets (of the same length) in parallel.
    """
    def __init__(self, datasets):
        """
        Dataset class iterating through multiple datasets of the same length.

        Arguments
        ---------
        datasets: sequence of datasets of same length

        """
        self.datasets = datasets
        self.num_inputs = len(self.datasets[0])

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        """
        Index the datasets and return the input + target of each dataset
        """
        outputs = []
        for dataset in self.datasets:
            outputs.append(dataset.__getitem__(index))
        return outputs


def make_dataset(df, scalar_encoder, transforms, test=False):
    """
    Does all the data manipulation and creates a dataset ready to be fed to the model
    :param df: pandas daraframe
    :param scalar_encoder: encoder to encode scalars
    :param transforms: data transformations
    :param test: indicate whether this a training/validation dataset or a test dataset
    :return: dataset
    """
    # Concat Bands into (N, 2, 75, 75) images
    band_1 = np.concatenate([im for im in df['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([im for im in df['band_2']]).reshape(-1, 75, 75)
    inc_angle = np.nan_to_num(df['inc_angle'].values)
    inc_angle = scalar_encoder.encode(inc_angle)
    # inc_angle_tr = np.zeros(train['inc_angle'].values.shape)
    full_img = np.stack([band_1, band_2], axis=1)

    # Dataset and DataLoader
    imgs = torch.from_numpy(full_img).float()
    angles = torch.from_numpy(inc_angle).float()
    if test:
        targets = None
    else:
        targets = torch.from_numpy(df['is_iceberg'].values).long()
    dataset_imgs = TensorDataset(imgs, targets, input_transform=transforms)
    dataset_angles = TensorDataset(angles, None)
    dataset = MultiDataset((dataset_imgs, dataset_angles))
    return dataset


def infer_ensemble(data, network_list):
    """Does a the forward pass for each network in the list `trial` number of times.
     Returns the avg and std of trials of all networks"""
    data_var_img = Variable(data[0][0].float().cuda())
    data_var_angle = Variable(data[1].float().cuda())
    networks_logits = []
    for net in network_list:
        trial_outputs = net(data_var_img, data_var_angle, trials=10).data
        networks_logits.append(trial_outputs)
    networks_logits = torch.stack(networks_logits, 1).squeeze_()
    probabilities = torch.sigmoid(networks_logits)
    pred_mean = torch.mean(probabilities)
    pred_std = torch.std(probabilities)
    return pred_mean, pred_std


def save_model(networks, dir):
    """Saves all models in a list"""
    save_dir = os.path.join(dir, "models")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, net in enumerate(networks):
        torch.save(net.state_dict(), os.path.join(save_dir, "net" + str(idx)+".pth"))


def load_model(dir):
    networks = []
    for filename in os.listdir(dir):
        net = Net()
        net.load_state_dict(torch.load(os.path.join(dir, filename)))
        net.train(True)
        net.cuda()
        networks.append(net)
    return networks


def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result