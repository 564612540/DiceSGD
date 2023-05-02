import torch
import math
from torch.optim import Optimizer
from fastDP import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from DiceSGD.optimizers_utils import PrivacyEngine_Dice

PRINT_FREQ = 25

def ClipSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr) 
    elif method == 'adam':
        optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise RuntimeError("Unknown Optimizer!")
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

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='mean')
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
            _, predicted = predict.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(t+1), 100.*correct/total, correct, total), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl, device, lr, logger)
    return model

def EFSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr) 
    elif method == 'adam':
        optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise RuntimeError("Unknown Optimizer!")
    privacy_engine = PrivacyEngine_Dice(
        model,
        batch_size=batch,
        sample_size=sample_size,
        epochs=epoch,
        noise_multiplier= 0.,
        max_grad_norm= C,
        loss_reduction= 'mean',
        clipping_fn='Abadi')
    privacy_engine.attach_dice(optimizer)
    acc_step = batch//minibatch


    # optimizer=torch.optim.SGD(model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='mean')
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
            _, predicted = predict.max(1)
            total += label.size(0)
            # actual_batch += label.size(0)
            correct += predicted.eq(label).sum().item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(t+1), 100.*correct/total, correct, total), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl,device, lr, logger)
    return model

def DPSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr) 
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

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='mean')
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
            _, predicted = predict.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(t+1), 100.*correct/total, correct, total), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl,device,  lr, logger)
    return model

def DiceSGD(model, train_dl, test_dl, batch, sample_size, minibatch, epoch, C, device, lr, method, logger):
    if method == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(), lr=lr) 
    elif method == 'adam':
        optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    else:
        raise RuntimeError("Unknown Optimizer!")
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch,
        sample_size=sample_size,
        epochs=epoch*int(math.log(epoch)+1),
        target_epsilon=2,
        max_grad_norm= C,
        loss_reduction= 'mean',
        clipping_fn='Abadi')
    privacy_engine.attach(optimizer)
    acc_step = batch//minibatch

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='mean')
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
            _, predicted = predict.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(t+1), 100.*correct/total, correct, total), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl,device,  lr, logger)
    return model
        
def test(epoch, t, model, test_dl, device,  lr, logger):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    model.train()
    print(" ")
    print('Epoch: ', epoch, ':', t+1, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    logger.update([lr,epoch],[100.*correct/total, test_loss/(batch_idx+1)])