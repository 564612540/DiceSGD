import torch
import math
from torch.optim import Optimizer
from fastDP import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from DiceSGD.optimizers_utils import PrivacyEngine_Dice

PRINT_FREQ = 25

def ClipSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    elif method == 'adam':
        optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise RuntimeError("Unknown Optimizer!")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, sample_size*epoch/minibatch)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch,
        sample_size=sample_size,
        epochs=epoch,
        noise_multiplier= 0.,
        max_grad_norm= C,
        loss_reduction= 'mean',
        clipping_fn='Abadi')
    privacy_engine.attach(optimizer)
    acc_step = batch//minibatch

    criterion = torch.nn.MSELoss(reduction='mean')
    # criterion = torch.nn.SmoothL1Loss(beta = 1e-5, reduction='mean')
    model.to(device)
    model.train()

    for E in range(epoch):
        train_loss = 0
        correct = 0
        total = 0
        for t, (input, label) in enumerate(train_dl):
            input = input.to(device)
            label = label.to(device)
            predict = model(input)
            loss = criterion(predict, label)
            loss.backward()

            train_loss += loss.item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Loss_Diff: %.3f'
                        % (train_loss/(t+1), train_loss/(t+1) - train_dl.dataset.sol["f"]), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl, device, lr, logger)
    return model

def EFSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C,C_2, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    elif method == 'adam':
        optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise RuntimeError("Unknown Optimizer!")
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, sample_size*epoch/minibatch, eta_min=lr*1e-4)
    privacy_engine = PrivacyEngine_Dice(
        model,
        batch_size=batch,
        sample_size=sample_size,
        epochs=epoch,
        noise_multiplier= 0.,
        max_grad_norm= C,
        error_max_grad_norm=C_2,
        loss_reduction= 'mean',
        clipping_fn='Abadi')
    privacy_engine.attach_dice(optimizer)
    acc_step = batch//minibatch


    # optimizer=torch.optim.SGD(model.parameters(), lr = lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    # criterion = torch.nn.SmoothL1Loss(beta = 1e-5, reduction='mean')
    model.to(device)
    model.train()

    for E in range(epoch):
        train_loss = 0
        correct = 0
        total = 0
        for t, (input, label) in enumerate(train_dl):
            input = input.to(device)
            label = label.to(device)
            predict = model(input)
            loss = criterion(predict, label)
            loss.backward()

            train_loss += loss.item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Loss_Diff: %.3f'
                        % (train_loss/(t+1), train_loss/(t+1) - train_dl.dataset.sol["f"]), end='')
        test(E, t, model, test_dl,device, lr, logger)
    return model

def DPSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95) 
    elif method == 'adam':
        optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise RuntimeError("Unknown Optimizer!")
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch,
        sample_size=sample_size,
        epochs=epoch,
        target_epsilon=2,
        max_grad_norm= C,
        loss_reduction= 'mean',
        clipping_fn='Abadi')
    privacy_engine.attach(optimizer)

    criterion = torch.nn.MSELoss(reduction='mean')
    model.to(device)
    model.train()
    acc_step = batch//minibatch

    for E in range(epoch):
        train_loss = 0
        correct = 0
        total = 0
        for t, (input, label) in enumerate(train_dl):
            input = input.to(device)
            label = label.to(device)
            predict = model(input)
            loss = criterion(predict, label)
            loss.backward()

            train_loss += loss.item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Loss_Diff: %.3f'
                        % (train_loss/(t+1), train_loss/(t+1) - train_dl.dataset.sol["f"]), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl,device,  lr, logger)
    return model

def DiceSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C, C_2, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95) 
    elif method == 'adam':
        optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise RuntimeError("Unknown Optimizer!")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, sample_size*epoch/minibatch)
    privacy_engine = PrivacyEngine_Dice(
        model,
        batch_size=batch,
        sample_size=sample_size,
        epochs=epoch* int(math.log(epoch)+1),
        target_epsilon=2,
        max_grad_norm= C,
        error_max_grad_norm = C_2,
        loss_reduction= 'mean',
        clipping_fn='Abadi')
    privacy_engine.attach_dice(optimizer)
    acc_step = batch//minibatch

    criterion = torch.nn.MSELoss(reduction='mean')
    model.to(device)
    model.train()

    for E in range(epoch):
        train_loss = 0
        correct = 0
        total = 0
        for t, (input, label) in enumerate(train_dl):
            input = input.to(device)
            label = label.to(device)
            predict = model(input)
            loss = criterion(predict, label)
            loss.backward()

            train_loss += loss.item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Loss_Diff: %.3f'
                        % (train_loss/(t+1), train_loss/(t+1) - train_dl.dataset.sol["f"]), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl,device,  lr, logger)
    return model
        
def test(epoch, t, model, test_dl, device,  lr, logger):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
    model.train()
    print(" ")
    print('Epoch: ', epoch, ':', t+1, 'Test Loss: %.3f | Loss_Diff: %.3f'
            % (test_loss/(len(test_dl.dataset)), test_loss/(len(test_dl.dataset)) - test_dl.dataset.sol["f"]))
    logger.update([lr,epoch],[test_loss/(len(test_dl.dataset)), test_loss/(len(test_dl.dataset)) - test_dl.dataset.sol["f"]])