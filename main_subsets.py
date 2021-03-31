import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from dataset import CIFAR10_loader
from architecture_new import ConvNet3
from train import custom_train, valid, valid_class
from utils import save_dict, _get_model_file
import os
import json
import argparse
from utils import aggregate_json_results
import time
from sklearn.model_selection import train_test_split
import numpy as np
FLAGS = None

def main():
    save_dir = os.path.join('output', "{model}").format(model=str(FLAGS.model_num))
    file_exists = os.path.isfile(_get_model_file(save_dir, str(FLAGS.model_num)))
    results_path = os.path.join(save_dir, 'results')
    if file_exists and not FLAGS.overwrite:
        print("Model file of \"%s\" already exists. Skipping training..." % FLAGS.model_num)
        results = aggregate_json_results(results_path)
        return results
    else:
        if file_exists:
            print("Model file exists, but will be overwritten...")

    transform_aug_train = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(FLAGS.resize_val),
         #transforms.RandomCrop(FLAGS.resize_val, padding=4),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_aug_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_train = transform_aug_train

    transform_test = transform_aug_test

    cif10 = CIFAR10_loader(root='./data/cifar-10-batches-py/', transform=transform_train, train=True)

    subset_idx, _ = train_test_split(np.arange(cif10.labels.shape[1]), train_size=FLAGS.subset_size, shuffle=True,
                                          stratify=cif10.labels.reshape((-1, 1)))
    train_idx, val_idx = train_test_split(subset_idx, train_size=0.8, shuffle=True,
                                  stratify=cif10.labels.reshape((-1, 1))[subset_idx])
    cif10_train = Subset(cif10, train_idx)
    print(len(cif10_train))
    cif10_val = Subset(cif10, val_idx)
    print(len(cif10_val))
    mytrainloader = DataLoader(cif10_train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)
    myvalloader = DataLoader(cif10_val, batch_size=FLAGS.batch_size, shuffle=False)
    cif10_test = CIFAR10_loader(root='./data/cifar-10-batches-py/', transform=transform_test, train=False)
    mytestloader = DataLoader(cif10_test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    convnet = ConvNet3()
    experiment_timer = time.time()
    results = custom_train(convnet, mytrainloader, valloader=myvalloader, testloader=mytestloader,
                           dir=save_dir, model_num=FLAGS.model_num,
                           epochs=FLAGS.epochs, device=device, lr=FLAGS.learning_rate)
    results['training_time'] = time.time() - experiment_timer
    flags_dict = vars(FLAGS)
    save_dict(os.path.join(save_dir, 'flags.json'), flags_dict)
    train_results = os.path.join(results_path, "train.json")
    save_dict(train_results, results)
    dict1 = valid(convnet, mytestloader, device=device)
    dict2 = valid_class(convnet, mytestloader, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
    final_dict = {**dict1, **dict2}
    acc_results = os.path.join(results_path, "test.json")
    save_dict(acc_results, final_dict)
    plt.plot(results['train_losses'])
    plt.title('Losses per epoch')
    plt.savefig(os.path.join(results_path, 'loss.png'))
    plt.close()
    plt.plot(results['train_scores'])
    plt.title('Accuracy per epoch')
    plt.savefig(os.path.join(results_path, 'acc.png'))
    plt.close()


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to run trainer.')
    parser.add_argument('--resize_val', type=int, default=32,
                        help='Resize value for images')
    parser.add_argument('--overwrite', type=bool, default=True,
                        help='If overwrite already trained model')
    parser.add_argument('--model_num', type=str, default="0",
                        help='Number of model')
    parser.add_argument('--subset_size', type=int, default=0.7,
                        help='Subset to use')
    FLAGS, unparsed = parser.parse_known_args()
    print_flags()
    main()
