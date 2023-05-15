import torch
from DiceSGD.trainers import ClipSGD, EFSGD, DPSGD, DiceSGD
from DiceSGD.dataset import generate_Cifar
import argparse
import timm
from opacus.validators import ModuleValidator
import warnings
import os
import gc

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
    parser.add_argument('--epoch', default=3, type=int,help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--mnbs', default=16, type=int, help='mini batch size')
    parser.add_argument('--C', default=0.5, nargs="+", type=float, help='clipping threshold')
    parser.add_argument('--C2', default=1.0, nargs="+", type=float, help='clipping threshold ration C2/C1')
    parser.add_argument('--algo', default='DiceSGD', type=str, help='algorithm (ClipSGD, EFSGD, DPSGD, DiceSGD)')
    parser.add_argument('--tag', default = '', type=str, help='log file tag')
    parser.add_argument('--model', default = 'vit_small_patch16_224', type=str, help='trained model')
    parser.add_argument('--method', default = 'sgd', type=str, help='sgd or adam')
    parser.add_argument('--log_path', default = './log', type=str, help='log file path')

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if not os.path.isdir(args.log_path):
            os.makedirs(args.log_path)
    log_file_path = '%s/%s_%s.csv'%(args.log_path,args.tag,args.algo)

    train_dl, test_dl = generate_Cifar(args.mnbs)
    sample_size = 50000
    for lr in args.lr:
        for C in args.C:
            for C2 in args.C2:
                lr_true = lr/C
                log_file = file_logger(log_file_path, 2, ["test_acc","test_loss"], heading = "E=%d, B=%d, C=%-.6f, C2=%-.6f, lr=%-.6f"%(args.epoch,args.bs,C, C2, lr))
                model = timm.create_model(args.model, pretrained=True, num_classes = 10)
                model = ModuleValidator.fix(model)
                for l,param in enumerate(model.parameters()):
                    if l<2:
                        param.requires_grad = False
                if args.algo == 'ClipSGD':
                    ClipSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, C, device, lr_true, args.method, log_file)
                elif args.algo == 'EFSGD':
                    EFSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, C, C2, device, lr_true, args.method, log_file)
                elif args.algo == 'DPSGD':
                    DPSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, C, device, lr_true, args.method, log_file)
                elif args.algo == 'DiceSGD':
                    DiceSGD(model, train_dl, test_dl, args.bs, sample_size, args.mnbs, args.epoch, C, C2, device, lr_true, args.method, log_file)
                else:
                    raise RuntimeError("Unknown Algorithm!")
                del model
                gc.collect()
                torch.cuda.empty_cache()

