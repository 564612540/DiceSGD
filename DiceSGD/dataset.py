import torch
import torchvision
from DiceSGD.img_utils import CIFAR10Policy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import warnings

def generate_Cifar(batchsize):
    trans_cifar = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trans_cifar_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar_train)
    dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)
    train_loader = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=False, pin_memory = True)
    test_loader = DataLoader(dataset_test,batch_size=batchsize*2,shuffle=False,drop_last=False, pin_memory = False)
    return train_loader, test_loader

def generate_GLUE(batchsize):
    pass