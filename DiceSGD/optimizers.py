import torch
import math
from torch.optim import Optimizer
from fastDP import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

PRINT_FREQ = 25
SPLIT_FACT = 2

def ClipSGD(model, train_dl, test_dl, batch, minibatch, epoch, C, device, lr, logger):
    optimizer_useless=torch.optim.SGD(model.parameters(), lr=lr) 
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch,
        sample_size=50000,
        epochs=epoch,
        noise_multiplier= 0.,
        max_grad_norm= C,
        loss_reduction= 'sum',
        clipping_fn='Abadi')
    privacy_engine.attach(optimizer_useless)
    acc_step = batch//minibatch

    optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    # optimizer=torch.optim.SGD(model.parameters(), lr = lr)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='sum')
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
                # for param in model.parameters():
                #     param.grad=param.summed_clipped_grad/batch
                #     del param.summed_clipped_grad
                for layer, param in enumerate(model.parameters()):
                    if param.requires_grad:
                        if hasattr(param, 'summed_clipped_grad') == False:
                            print(layer)
                            param.grad = param.grad/batch
                        else:
                            param.grad=param.summed_clipped_grad/batch
                            del param.summed_clipped_grad
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(t+1), 100.*correct/total, correct, total), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl, device, lr, logger)
    return model

def EFSGD(model, train_dl, test_dl, batch, minibatch, epoch, C, device, lr, logger):
    C1 = C/SPLIT_FACT
    C2 = C/SPLIT_FACT
    optimizer_useless=torch.optim.SGD(model.parameters(), lr= lr) 
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch,
        sample_size=50000,
        epochs=epoch,
        noise_multiplier= 0.,
        max_grad_norm= C1,
        loss_reduction= 'sum',
        clipping_fn='Abadi')
    privacy_engine.attach(optimizer_useless)
    acc_step = batch//minibatch

    optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    # optimizer=torch.optim.SGD(model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='sum')
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
                error_norms = []
                for param in model.parameters():
                    if param.requires_grad:
                        if hasattr(param,'error'):
                            first_minibatch = False
                            error_norms.append(param.error.norm(2))
                        else:
                            # param.error = None
                            first_minibatch = True
                            error_norms.append(torch.tensor(0.))
                error_norm = torch.stack(error_norms).norm(2)
                if error_norm<1e-6:
                    error_norm = 1e-6
                # print(error_norm)
                for param in model.parameters():
                    if param.requires_grad:
                        grad_diff=(param.grad-param.summed_clipped_grad)/batch
                        param.grad=param.summed_clipped_grad/batch
                        del param.summed_clipped_grad
                        if first_minibatch:
                            param.error=grad_diff
                        else:
                            param.grad += param.error*torch.clamp_max(C2/error_norm,1.)
                            # param.error=param.error*(1-torch.clamp_max(C2/error_norm,1.))+grad_diff
                            param.error = grad_diff
                        del grad_diff
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(t+1), 100.*correct/total, correct, total), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl,device, lr, logger)
    return model

def DPSGD(model, train_dl, test_dl, batch, minibatch, epoch, C, device, lr, logger):
    optimizer_useless=torch.optim.SGD(model.parameters(), lr=lr) 
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch,
        sample_size=50000,
        epochs=epoch,
        target_epsilon=2,
        max_grad_norm= C,
        loss_reduction= 'sum',
        clipping_fn='Abadi')
    privacy_engine.attach(optimizer_useless)

    sigma = get_noise_multiplier(
        target_delta=1e-5,
        target_epsilon=2,
        sample_rate=batch/50000,
        epochs=epoch
    )

    optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer=torch.optim.SGD(model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='sum')
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
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad=param.summed_clipped_grad/batch + torch.normal(
                            mean=0,
                            std=sigma * C,
                            size=param.size(),
                            device=device,
                        )/batch
                        del param.summed_clipped_grad
                optimizer.step()
                optimizer.zero_grad()

            if t==0 or (t+1)%PRINT_FREQ == 0 or ((t + 1) == len(train_dl)):
                print('\rEpoch: ', E, ':', t+1, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(t+1), 100.*correct/total, correct, total), end='')
                # correct = 0
                # total = 0
        test(E, t, model, test_dl,device,  lr, logger)
    return model

def DiceSGD(model, train_dl, test_dl, batch, minibatch, epoch, C, device, lr, logger):
    C1 = C/(SPLIT_FACT*batch)
    C2 = C/(SPLIT_FACT*batch)
    optimizer_useless=torch.optim.SGD(model.parameters(), lr=lr) 
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch,
        sample_size=50000,
        epochs=epoch,
        target_epsilon=2,
        max_grad_norm= C1,
        loss_reduction= 'sum',
        clipping_fn='Abadi')
    privacy_engine.attach(optimizer_useless)
    acc_step = batch//minibatch

    sigma = get_noise_multiplier(
        target_delta=1e-5,
        target_epsilon=2,
        sample_rate=batch/50000,
        epochs=epoch
    )

    optimizer=torch.optim.Adam(model.parameters(), lr = lr)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer=torch.optim.SGD(model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1,reduction='sum')
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
                error_norms = []
                for param in model.parameters():
                    if param.requires_grad:
                        if hasattr(param,'error'):
                            first_minibatch = False
                            error_norms.append(param.error.norm(2))
                        else:
                            # param.error = None
                            first_minibatch = True
                            error_norms.append(torch.tensor(0.))
                error_norm = torch.stack(error_norms).norm(2) + 1e-6

                for param in model.parameters():
                    if param.requires_grad:
                        grad_diff=(param.grad-param.summed_clipped_grad)/batch
                        param.grad=param.summed_clipped_grad/batch + torch.normal(
                            mean=0,
                            std=sigma* C1,
                            size=param.size(),
                            device=device,
                        )/batch
                        del param.summed_clipped_grad
                        if first_minibatch:
                            param.error=grad_diff
                        else:
                            param.grad += param.error*torch.clamp_max(C2/error_norm,1.) + torch.normal(
                                mean=0,
                                std=sigma * C2,
                                size=param.size(),
                                device=device,
                            )
                            # param.error=param.error*(1-torch.clamp_max(C2/error_norm,1.))+grad_diff
                            param.error=grad_diff
                        del grad_diff
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