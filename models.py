import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        drop_p = 0.25
        drop_p_fc = 0.5
        self.batch = nn.BatchNorm2d(2)
        self.conv00 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv01 = nn.Conv2d(2, 16, kernel_size=5, padding=2)
        self.dout0 = nn.Dropout2d(drop_p)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.dout1 = nn.Dropout2d(drop_p)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dout2 = nn.Dropout2d(drop_p_fc)
        self.fc1 = nn.Linear(64 * 9 * 9 + 100, 1000)
        self.dout3 = nn.Dropout(drop_p_fc)
        self.fc2 = nn.Linear(1000, 200)
        self.dout4 = nn.Dropout(drop_p_fc)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x, angle, trials=1):
        if trials > 1:
            # repeating each sample `trials` times to get confidence intervals by applying dropout.
            self.train()  # model needs to be in training mode for dropout layers to work
            if len(x.size()) == 3:
                x.data.unsqueeze_(0)
            x = torch.stack([x] * trials, 1).view(x.size()[0] * trials, 2, 75, 75)
            angle = torch.stack([angle] * trials, 1).view(angle.size()[0] * trials, 100)

        x = self.batch(x)
        x = self.pool(F.relu(self.dout0(torch.cat((self.conv00(x), self.conv01(x)), 1))))
        x = self.pool(F.relu(self.dout1(torch.cat((self.conv10(x), self.conv11(x)), 1))))
        x = self.pool(F.relu(self.dout2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, angle), 1)
        x = F.leaky_relu(self.dout3(self.fc1(x)))
        x = F.leaky_relu(self.dout4(self.fc2(x)))
        x = self.fc3(x)
        return x


class CNNClassifier(nn.Module):
    dropout = [0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1]
    def __init__(self, img_size, img_ch, kernel_size, pool_size, n_out):
        super(CNNClassifier, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.n_out = n_out
        self. sig =torch.nn.Sigmoid()
        self.all_losses = []
        self.val_losses = []

        self.build_model()
        print (self)
    # end constructor

    def build_model(self):
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.img_ch, 16, kernel_size=self.kernel_size, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.Dropout2d(p=dropout[0]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(self.pool_size)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=self.kernel_size, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(p=dropout[1]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(self.pool_size)
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.Dropout2d(p=dropout[2]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(self.pool_size)
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=self.kernel_size -1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout2d(p=dropout[3]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(self.pool_size)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 32, kernel_size=self.kernel_size - 1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout2d(p=dropout[4]),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(self.pool_size)
        )

        self.fc = torch.nn.Linear(288, self.n_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.shrink(x)
        x = self.fc(x)
        return self.sig(x)

    # end method forward

    def shrink(self, X):
        return X.view(X.size(0), -1)
