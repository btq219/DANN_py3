import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils.data_loader import get_dataset
from model.model_transformer import DANN
from utils.utils import split_dataset, optimizer_scheduler
from test import test
from torch.utils.tensorboard import SummaryWriter

source_dataset_name = 'sunny'
target_dataset_name = 'snowy'
source_dataset_root = os.path.join('dataset/', source_dataset_name)
target_dataset_root = os.path.join('dataset/', target_dataset_name)
model_root = 'exp_trans/models'

# add summeryWriter
train_writer = SummaryWriter(log_dir=model_root + "/tb/train")
test_writer = SummaryWriter(log_dir=model_root + "/tb/test")

cuda = True
cudnn.benchmark = True
lr = 1e-4
theta = 1
batch_size = 32
n_epoch = 100

# manual_seed = random.randint(1, 10000)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)

# load model
my_net = DANN()

# setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=lr)

# load data
source_data_files = [os.path.join(source_dataset_root, file) for file in os.listdir(source_dataset_root) if file.endswith('.mat')]

dataset_source = get_dataset(source_data_files)
dataset_source_train, dataset_source_test = split_dataset(dataset_source, train_size=0.8)
dataloader_source_train = DataLoader(dataset_source_train, batch_size=batch_size, shuffle=True)
dataloader_source_test = DataLoader(dataset_source_test, batch_size=batch_size, shuffle=True)

target_data_files = [os.path.join(target_dataset_root, file) for file in os.listdir(target_dataset_root) if file.endswith('.mat')]

dataset_target = get_dataset(target_data_files)
dataset_target_train, dataset_target_test = split_dataset(dataset_target, train_size=0.8)
dataloader_target_train = DataLoader(dataset_target_train, batch_size=batch_size, shuffle=True)
dataloader_target_test = DataLoader(dataset_target_test, batch_size=batch_size, shuffle=True)



loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True


# training
best_accu_t = 0.0
for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source_train), len(dataloader_target_train))
    data_source_iter = iter(dataloader_source_train)
    data_target_iter = iter(dataloader_target_train)


    n_total = 0
    n_class_correct = 0
    n_domain_correct = 0
    n_err_label = 0
    n_err_domain = 0


    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.__next__()
        s_img, s_label = data_source

        optimizer, lr_current = optimizer_scheduler(optimizer=optimizer, lr=lr,p=p)
        optimizer.zero_grad()

        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()


        class_output, domain_output = my_net(x=s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label.long())
        err_s_domain = loss_domain(domain_output, domain_label)

        ####### recording results #############
        pred_class = class_output.data.max(1, keepdim=True)[1]
        n_class_correct += pred_class.eq(s_label.data.view_as(pred_class)).cpu().sum()
        pred_domain = domain_output.data.max(1, keepdim=True)[1]
        n_domain_correct += pred_domain.eq(domain_label.data.view_as(pred_domain)).cpu().sum()
        n_total += batch_size
        n_err_label += loss_class(class_output, s_label.long()).cpu()
        n_err_domain += loss_domain(domain_output, domain_label).cpu()


        # training model using target data
        data_target = data_target_iter.__next__()
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = my_net(x=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)

        # pred_domain = domain_output.data.max(1, keepdim=True)[1]
        # n_domain_correct += pred_domain.eq(domain_label.data.view_as(pred_domain)).cpu().sum()

        err = err_s_label + theta * (err_t_domain + err_s_domain)
        err.backward()
        optimizer.step()


        ############ training process ##############
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        torch.save(my_net,
                   '{0}/model_epoch_current.pth'.format(model_root))

        train_writer.add_scalar(tag="loss_training/all", scalar_value=err, global_step=epoch * len_dataloader + i)
        train_writer.add_scalar(tag="loss_training/domain", scalar_value=err_t_domain + err_s_domain, global_step=epoch * len_dataloader + i)
        train_writer.add_scalar(tag="loss_training/class", scalar_value=err_s_label, global_step=epoch * len_dataloader + i)
        train_writer.add_scalar(tag="params/alpha", scalar_value=alpha, global_step=epoch * len_dataloader + i)
        train_writer.add_scalar(tag="params/lr_current", scalar_value=lr_current, global_step=epoch * len_dataloader + i)

    #################### training summary in one epoch ############################
    print('\n')
    accu_s_class, accu_s_domain, _, _ = test(dataloader_source_test, False, model_root)
    print('Accuracy of the %s dataset: %f' % ('sunny', accu_s_class))
    accu_t_class, accu_t_domain, err_t_label, err_t_domain = test(dataloader_target_test, True, model_root)
    print('Accuracy of the %s dataset (Class): %f' % ('snowy', accu_t_class))
    print('Accuracy of the %s dataset (Domain): %f' % ('snowy', accu_t_domain))
    print('Class Loss of the %s dataset: %f' % ('snowy', err_t_label))
    print('Domain Loss of the %s dataset: %f\n' % ('snowy', err_t_domain))


    acc_class = n_class_correct.data.numpy() * 1.0 / n_total
    acc_domain = n_domain_correct.data.numpy() * 1.0 / n_total
    err_label = n_err_label.data.numpy() * 1.0 / len_dataloader
    err_domain = n_err_domain.data.numpy() * 1.0 / len_dataloader

    train_writer.add_scalar(tag="Loss/class", scalar_value=err_label, global_step=epoch)
    train_writer.add_scalar(tag="Loss/domain", scalar_value=err_domain,global_step=epoch)

    train_writer.add_scalar(tag="Acc/class", scalar_value=acc_class, global_step=epoch)
    train_writer.add_scalar(tag="Acc/domain", scalar_value=acc_domain,global_step=epoch)
    # train_writer.add_scalar(tag="loss_train/all", scalar_value=err, global_step=epoch * len_dataloader + i)
    # train_writer.add_scalar(tag="params/lr", scalar_value=lr, global_step=epoch * len_dataloader + i)


    # test_writer.add_scalar(tag="acc_val/source_class", scalar_value=accu_s_class, global_step=epoch)
    # test_writer.add_scalar(tag="acc_val/source_domain", scalar_value=accu_s_domain, global_step=epoch)
    test_writer.add_scalar(tag="Acc/class", scalar_value=accu_t_class, global_step=epoch )
    test_writer.add_scalar(tag="Acc/domain", scalar_value=accu_t_domain, global_step=epoch)
    test_writer.add_scalar(tag="Loss/domain", scalar_value=err_t_domain, global_step=epoch)
    test_writer.add_scalar(tag="Loss/class", scalar_value=err_t_label, global_step=epoch)

    if accu_t_class > best_accu_t:
        best_accu_s = accu_s_class
        best_accu_t = accu_t_class
        torch.save(my_net, '{0}/{1}_{2}_model_epoch_best.pth'.format(model_root, source_dataset_name, target_dataset_name))

print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % (source_dataset_name, best_accu_s))
print('Accuracy of the %s dataset: %f' % (target_dataset_name, best_accu_t))
print('Corresponding model was save in {0}/{1}_{2}_model_epoch_best.pth'
      .format(model_root, source_dataset_name, target_dataset_name))
