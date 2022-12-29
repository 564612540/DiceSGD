import torch
import torchvision
from optimizers import ClipSGD, EFSGD, DPSGD, DiceSGD
from img_utils import CIFAR10Policy
from model_utils import create_resnet
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from opacus.validators import ModuleValidator

def generate_Cifar(batchsize):
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trans_cifar_train = transforms.Compose([transforms.RandomCrop(size=32, padding = 4), transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    # trans_cifar = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    # trans_cifar_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(size=224, padding = 4), transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar_train)
    dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)
    train_loader = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=False, pin_memory = True)
    test_loader = DataLoader(dataset_test,batch_size=batchsize*4,shuffle=True,drop_last=False, pin_memory = False)
    return train_loader, test_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=3, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--mnbs', default=16, type=int, help='mini batch size')
    parser.add_argument('--C', default=0.5, type=float, help='clipping threshold')
    parser.add_argument('--algo', default='DiceSGD', type=str, help='algorithm (ClipSGD, EFSGD, DPSGD, DiceSGD)')

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    train_dl, test_dl = generate_Cifar(args.mnbs)
    # model = timm.create_model('resnet18', pretrained=True, num_classes = 10)
    model = create_resnet(num_classes=10)
    model = ModuleValidator.fix(model)
    # for p in model.parameters():
    #     print(p.size())
    if args.algo == 'ClipSGD':
        ClipSGD(model, train_dl, test_dl, args.bs, args.mnbs, args.epoch, args.C, device, args.lr)
    elif args.algo == 'EFSGD':
        EFSGD(model, train_dl, test_dl, args.bs, args.mnbs, args.epoch, args.C, device, args.lr)
    elif args.algo == 'DPSGD':
        DPSGD(model, train_dl, test_dl, args.bs, args.mnbs, args.epoch, args.C, device, args.lr)
    elif args.algo == 'DiceSGD':
        DiceSGD(model, train_dl, test_dl, args.bs, args.mnbs, args.epoch, args.C, device, args.lr)
    else:
        raise RuntimeError("Unknown Algorithm!")
