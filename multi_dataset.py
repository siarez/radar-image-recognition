from torch.utils.data import dataset


class MultiDataset(dataset.Dataset):
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
