from torch.utils.data import Dataset, DataLoader
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        mydict = pickle.load(fo, encoding='bytes')
    return mydict


class CIFAR10_loader(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = list()
        self.labels = list()
        self.transform = transform
        # Load data
        if train:
            for i in range(1, 6):
                path = root + f'data_batch_{i}'
                mydict = unpickle(path)
                for i in range(len(mydict[b'data'])):
                    self.data.append(mydict[b'data'][i].reshape(3, 32, 32))
                self.labels = [*self.labels, mydict[b'labels']]
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            self.labels = np.reshape(self.labels, (1, -1))
            # self.data = self.data/255
            # self.data = self.data.astype(np.float32)
            self.max_labels = np.max(self.labels)
        else:
            path = root + 'test_batch'
            mydict = unpickle(path)
            self.data = mydict[b'data']
            self.labels = mydict[b'labels']
            self.data = np.reshape(self.data, (-1, 3, 32, 32))
            self.labels = np.array(self.labels)
            self.labels = np.reshape(self.labels, (1, -1))
            # self.data /= 255
            # self.data = self.data.astype(np.float32)
            self.max_labels = np.max(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label = self.labels[0, item]
        img = self.data[item]
        # img = np.reshape(img, (3,32,32))
        target = np.zeros(self.max_labels)
        target[label - 1] = 1
        img = np.transpose(img, (1, 2, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, label