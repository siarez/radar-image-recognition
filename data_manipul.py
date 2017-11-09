import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import mixture
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsample import transforms, TensorDataset
from models import Net, CNNClassifier


"""
This files includes little scribbles of code to plot and analyzed the data
"""


data = pd.read_json('data/train.json')
#test = pd.read_json('test.json')
data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
#test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')


def fit_gmm(data):
    bic = []
    lowest_bic = np.infty
    for n_comp in range(1, 3):
        gmm = mixture.GaussianMixture(n_components=n_comp)
        gmm.fit(np.swapaxes(data, 0, 1))
        bic.append(gmm.bic(np.swapaxes(data, 0, 1)))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    return best_gmm

def create_minis(data):
    data = pd.read_json('train.json')
    data.drop(data.index[:1600], inplace=True)
    data.to_json('train-mini.json')


def plot_sample(sample):
    sample = sample.numpy()
    c = ( 'Not Hotdog', 'Hotdog')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(sample[0].reshape(75, 75))
    ax2.imshow(sample[1].reshape(75, 75))
    ax3.hist(sample[0].ravel(), bins=256, fc='k', ec='k');
    ax4.hist(sample[1].ravel(), bins=256, fc='k', ec='k');
    f.set_figheight(10)
    f.set_figwidth(10)
    #plt.suptitle(c[df['is_iceberg'].iloc[idx]])
    plt.show()

class Hist:
    def __init__(self, num_of_bins, min, max):
        self.num_bins = num_of_bins
        self.min = min
        self.max = max
        self.multiplier = (self.num_bins - 2)  / (self.max - self.min)
        self.lookup = np.eye(self.num_bins)

    def put_in_bin(self, input_arr):
        scaled_input = np.array((input_arr - self.min) * self.multiplier).astype(np.int)
        scaled_input = np.minimum(scaled_input, self.num_bins - 1)
        scaled_input = np.maximum(scaled_input, 0)
        return self.lookup[scaled_input, :]


hist = Hist(100, 30, 45)



band_1_tr = np.concatenate([im for im in data['band_1']]).reshape(-1, 75, 75)
band_2_tr = np.concatenate([im for im in data['band_2']]).reshape(-1, 75, 75)
inc_angle_tr = np.nan_to_num(data['inc_angle'].values)
encoded_angle = hist.put_in_bin(inc_angle_tr)
hist_test = np.sum(encoded_angle, 0)
print(hist_test.shape)
#inc_angle_test = np.nan_to_num(test['inc_angle'].values)
target = np.array(data['is_iceberg'].values)
# input, bins=100, min=0, max=0, out=None) → Tensor
plt.scatter(np.arange(0, hist.num_bins), hist_test)
#plt.scatter(inc_angle_tr, target)
#plt.xlim(0, 48)
#plt.hist(inc_angle_test, bins=100)
#plt.hist(inc_angle_tr, bins=100)

plt.show()

del data
full_img_tr = np.stack([band_1_tr, band_2_tr], axis=1)

my_transforms = transforms.RandomAffine(rotation_range=180,
                 translation_range=0.2,
                 shear_range=None,
                 zoom_range=(0.8, 1.2))

my_transforms = transforms.Compose(my_transforms.transforms)
test_imgs = torch.from_numpy(full_img_tr).float().cuda()
test_dataset = TensorDataset(test_imgs, input_transform=my_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("loader len:", len(test_loader))

while(True):
    #index = np.random.randint(0, len(data), 1)
    index = 0
    for _data in tqdm(test_loader, total=len(test_loader)):
        print(index)
        plot_sample(_data.squeeze_().cpu())
        index += 1



    best_gmm = fit_gmm(band_1_tr[index])
    print("means: ", best_gmm.means_)
    print("covs: ", best_gmm.covariances_)
    print("weights: ", best_gmm.weights_)



