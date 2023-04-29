import torch
from DiceSGD.optimizers import ClipSGD, EFSGD, DPSGD, DiceSGD
from DiceSGD.model_utils import create_resnet, create_cnn
from DiceSGD.dataset import generate_Cifar, generate_Cifar100
import argparse
import timm
from opacus.validators import ModuleValidator
import warnings

# def generate_Cifar(batchsize):
#     trans_cifar = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
#     trans_cifar_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
#     dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar_train)
#     dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)
#     train_loader = DataLoader(dataset_train,batch_size=batchsize,shuffle=True,drop_last=False, pin_memory = True)
#     test_loader = DataLoader(dataset_test,batch_size=batchsize*2,shuffle=False,drop_last=False, pin_memory = False)
#     return train_loader, test_loader

class file_logger():
    def __init__(self, path, time_num, item_list, heading = None):
        head = ['time_'+str(i) for i in range(time_num)]
        head_str = ','.join(head)+','+','.join(item_list)
        self.path = path
        self.time_num = time_num
        self.item_length = len(item_list)
        with open(self.path,'a') as fp:
            if heading is not None:
                print(heading, file=fp)
            print(head_str, file=fp)
    
    def update(self, time_list, item_list):
        if len(time_list)!=self.time_num or len(item_list)!=self.item_length:
            raise RuntimeError('incorrect log information')
        log_info = ','.join(map(str,time_list))+','+','.join(map(str,item_list))
        with open(self.path,'a') as fp:
            print(log_info, file=fp)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.1, nargs="+", type=float, help='learning rate list')
    parser.add_argument('--epoch', default=3, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--mnbs', default=16, type=int, help='mini batch size')
    parser.add_argument('--C', default=0.5, type=float, help='clipping threshold')
    parser.add_argument('--algo', default='DiceSGD', type=str, help='algorithm (ClipSGD, EFSGD, DPSGD, DiceSGD)')
    parser.add_argument('--tag', default = '', type=str, help='log file tag')
    parser.add_argument('--model', default = 'vit_small_patch16_224', type=str, help='trained model')

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    log_file_path = './log/%s_%s_%-.3f.csv'%(args.tag,args.algo,args.C)
    log_file = file_logger(log_file_path, 2, ["test_acc","test_loss"], heading = "E=%d, B=%d, C=%-.4f"%(args.epoch,args.bs,args.C))

    train_dl, test_dl = generate_Cifar100(args.mnbs)
    sample_size = 50000
    for lr in args.lr:
        model = timm.create_model(args.model, pretrained=True, num_classes = 100)
        model = ModuleValidator.fix(model)
        for l,param in enumerate(model.parameters()):
            if l<2:
                param.requires_grad = False
        if args.algo == 'ClipSGD':
            ClipSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, args.C, device, lr, log_file)
        elif args.algo == 'EFSGD':
            EFSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, args.C, device, lr, log_file)
        elif args.algo == 'DPSGD':
            DPSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, args.C, device, lr, log_file)
        elif args.algo == 'DiceSGD':
            DiceSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, args.C, device, lr, log_file)
        else:
            raise RuntimeError("Unknown Algorithm!")
