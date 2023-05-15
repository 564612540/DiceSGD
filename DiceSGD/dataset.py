import os
import os.path
import pickle
import torch
import torchvision
from DiceSGD.img_utils import CIFAR10Policy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
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

def generate_Cifar100(batchsize):
    trans_cifar = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trans_cifar_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar_train)
    dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar)
    train_loader = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=False, pin_memory = True)
    test_loader = DataLoader(dataset_test,batch_size=batchsize*2,shuffle=False,drop_last=False, pin_memory = False)
    return train_loader, test_loader

def generate_synthetic(batchsize):
    dataset_train = SyntheticData('./data/synthetic', dim = 100)
    dataset_test = SyntheticData('./data/synthetic', dim = 100)
    train_loader = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=False, pin_memory = True)
    test_loader = DataLoader(dataset_test,batch_size=batchsize*2,shuffle=False,drop_last=False, pin_memory = False)
    return train_loader, test_loader

class SyntheticData(Dataset):
    def __init__(self, root, dim = 100) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        fpath = os.path.join(root, "data")
        if os.path.isfile(fpath):
            print("Files already downloaded and verified")
        else:
            self.generate_random_data(dim, fpath)
        self.data, self.target, self.sol = self.load_data(fpath)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        x = self.data[index]
        y = self.target[index]
        return x,y

    def generate_random_data(self, dim, fpath):
        size = dim * 10
        w = torch.randn(dim)
        b = torch.randn(1)
        X = torch.randn(size, dim)
        noise = torch.randn(size, 1)*0.1 + torch.randint(1,10,(size,1))*0.1
        y = torch.matmul(X, w.reshape((-1, 1))) + b + noise
        X1 = torch.hstack([X, torch.ones(size,1)])
        sol = torch.linalg.solve(torch.matmul(X1.T, X1), torch.matmul(X1.T, y))
        w_opt = sol[:-1]
        b_opt = sol[-1]
        f_opt = torch.nn.functional.mse_loss(torch.matmul(X1, sol) , y)
        data = {"x":X, "y":y, "w_opt": w_opt, "b_opt": b_opt, "f_opt":f_opt}
        with open(fpath, mode="wb") as fp:
            pickle.dump(data,fp)
        
    def load_data(self, fpath):
        with open(fpath, mode="rb") as fp:
            data = pickle.load(fp)
        x = data['x']
        y = data['y']
        w_opt = data['w_opt']
        b_opt = data['b_opt']
        f_opt = data['f_opt']
        sol = {"w":w_opt, "b":b_opt, "f":f_opt.item()}
        return x,y,sol

