import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils.data_loader import get_dataset
from model.modelV2 import DANN, LabelPredictor, DomainClassifier, FeatureExtractorLeakly
from utils.utils import split_dataset, optimizer_scheduler
from test import test
from torch.utils.tensorboard import SummaryWriter

source_dataset_name = 'sunny'
target_dataset_name = 'snowy'
source_dataset_root = os.path.join('dataset/8positions_200/', source_dataset_name)
target_dataset_root = os.path.join('dataset/8positions_200/', target_dataset_name)
model_root = 'exp_conv/models_snowy/'

# add summeryWriter
train_writer = SummaryWriter(log_dir=model_root + "tb/train")
test_writer = SummaryWriter(log_dir=model_root + "tb/test")

cuda = True
cudnn.benchmark = True
lr = 1e-3
theta = 1
batch_size = 32
n_epoch = 200
alpha = 0.1

# manual_seed = random.randint(1, 10000)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)

# load model
# my_net = DANN()
feature_extractor = FeatureExtractorLeakly()
class_predictor = LabelPredictor()
domain_classifier = DomainClassifier()

# setup optimizer
optim_F = optim.Adam(feature_extractor.parameters(), lr=lr)
optim_C = optim.Adam(class_predictor.parameters(), lr=lr)
optim_D = optim.Adam(domain_classifier.parameters(), lr=lr)

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
    feature_extractor = feature_extractor.cuda()
    class_predictor = class_predictor.cuda()
    domain_classifier = domain_classifier.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

# for p in my_net.parameters():
#     p.requires_grad = True


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

        data_source = data_source_iter.__next__()
        s_img, s_label = data_source
        data_target = data_target_iter.__next__()
        t_img, _ = data_target

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            t_img = t_img.cuda()

        domain_label_s = torch.zeros(batch_size).long()
        domain_label_t = torch.ones(batch_size).long()
        domain_labels = torch.cat([domain_label_s, domain_label_t], dim=0).cuda()

        # training domain classifier
        inputs = torch.cat([s_img, t_img], dim=0)
        features = feature_extractor(inputs)
        domain_outputs = domain_classifier(features.detach())

        loss_D = loss_domain(domain_outputs, domain_labels)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()


        # training class predictor and feature extractor
        if i % 5 == 0:
            class_output = class_predictor(feature_extractor(s_img))
            domain_outputs = domain_classifier(features)
            loss_D = loss_domain(domain_outputs, domain_labels)

            loss = loss_class(class_output, s_label.long()) - alpha * loss_D
            loss_C = loss_class(class_output, s_label.long())

            optim_F.zero_grad()
            optim_C.zero_grad()
            optim_D.zero_grad()
            loss.backward()
            optim_C.step()
            optim_F.step()

        #
        # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        #
        # # training model using source data
        # data_source = data_source_iter.__next__()
        # s_img, s_label = data_source
        # data_target = data_target_iter.__next__()
        # t_img, _ = data_target
        #
        # optimizer, lr_current = optimizer_scheduler(optimizer=optimizer, lr=lr,p=p)
        # optimizer.zero_grad()
        #
        # batch_size = len(s_label)
        #
        # domain_label = torch.zeros(batch_size).long()
        #
        # if cuda:
        #     s_img = s_img.cuda()
        #     s_label = s_label.cuda()
        #     domain_label = domain_label.cuda()
        #
        #
        # class_output, domain_output = my_net(x=s_img, alpha=alpha)
        # err_s_label = loss_class(class_output, s_label.long())
        # err_s_domain = loss_domain(domain_output, domain_label)
        #
        # ####### recording results #############
        # pred_class = class_output.data.max(1, keepdim=True)[1]
        # n_class_correct += pred_class.eq(s_label.data.view_as(pred_class)).cpu().sum()
        # pred_domain = domain_output.data.max(1, keepdim=True)[1]
        # n_domain_correct += pred_domain.eq(domain_label.data.view_as(pred_domain)).cpu().sum()
        # n_total += batch_size
        # n_err_label += loss_class(class_output, s_label.long()).cpu()
        # n_err_domain += loss_domain(domain_output, domain_label).cpu()
        #
        #
        # # training model using target data
        # data_target = data_target_iter.__next__()
        # t_img, _ = data_target
        #
        # batch_size = len(t_img)
        #
        # domain_label = torch.ones(batch_size).long()
        #
        # if cuda:
        #     t_img = t_img.cuda()
        #     domain_label = domain_label.cuda()
        #
        # _, domain_output = my_net(x=t_img, alpha=alpha)
        # err_t_domain = loss_domain(domain_output, domain_label)
        #
        # # pred_domain = domain_output.data.max(1, keepdim=True)[1]
        # # n_domain_correct += pred_domain.eq(domain_label.data.view_as(pred_domain)).cpu().sum()
        #
        # err = err_s_label + theta * (err_t_domain + err_s_domain)
        # err.backward()
        # optimizer.step()
        #

        ############ training process ##############
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], loss_label: %f, loss_domain: %f' \
              % (epoch, i + 1, len_dataloader, loss_C.data.cpu().numpy(), loss_D.data.cpu().numpy()))
        sys.stdout.flush()
        torch.save(feature_extractor,'{0}/feature_extractor.pth'.format(model_root))
        torch.save(class_predictor,'{0}/class_predictor.pth'.format(model_root))
        torch.save(domain_classifier,'{0}/domain_classifier.pth'.format(model_root))

        train_writer.add_scalar(tag="loss_training/loss_C", scalar_value=loss_C, global_step=epoch * len_dataloader + i)
        train_writer.add_scalar(tag="loss_training/loss_D", scalar_value=loss_D, global_step=epoch * len_dataloader + i)


        # train_writer.add_scalar(tag="loss_training/all", scalar_value=err, global_step=epoch * len_dataloader + i)
        # train_writer.add_scalar(tag="loss_training/domain", scalar_value=err_t_domain + err_s_domain, global_step=epoch * len_dataloader + i)
        # train_writer.add_scalar(tag="loss_training/class", scalar_value=err_s_label, global_step=epoch * len_dataloader + i)
        # train_writer.add_scalar(tag="params/alpha", scalar_value=alpha, global_step=epoch * len_dataloader + i)
        # train_writer.add_scalar(tag="params/lr_current", scalar_value=lr_current, global_step=epoch * len_dataloader + i)


    #################### training summary in one epoch ############################
    feature_extractor.eval()
    class_predictor.eval()
    with torch.no_grad():
        corrects = torch.zeros(1).cuda()
        for idx, (src, labels) in enumerate(dataloader_source_test):
            src, labels = src.cuda(), labels.cuda()
            c = class_predictor(feature_extractor(src))
            _, preds = torch.max(c, 1)
            corrects += (preds == labels).sum()
        acc = corrects.item() / len(dataloader_source_test.dataset)
        print('\r')
        print('***** Eval Result: {:.4f}, Step: {}'.format(acc, epoch))

        corrects = torch.zeros(1).cuda()
        for idx, (tgt, labels) in enumerate(dataloader_target_test):
            tgt, labels = tgt.cuda(), labels.cuda()
            c = class_predictor(feature_extractor(tgt))
            _, preds = torch.max(c, 1)
            corrects += (preds == labels).sum()
        acc = corrects.item() / len(dataloader_target_test.dataset)
        print('***** Test Result: {:.4f}, Step: {}'.format(acc, epoch))
        # acc_lst.append(acc)

    feature_extractor.train()
    class_predictor.train()




#     print('\n')
#     accu_s_class, accu_s_domain, _, _ = test(dataloader_source_test, False, model_root)
#     print('Accuracy of the %s dataset: %f' % ('sunny', accu_s_class))
#     accu_t_class, accu_t_domain, err_t_label, err_t_domain = test(dataloader_target_test, True, model_root)
#     print('Accuracy of the %s dataset (Class): %f' % ('snowy', accu_t_class))
#     print('Accuracy of the %s dataset (Domain): %f' % ('snowy', accu_t_domain))
#     print('Class Loss of the %s dataset: %f' % ('snowy', err_t_label))
#     print('Domain Loss of the %s dataset: %f\n' % ('snowy', err_t_domain))
#
#
#     acc_class = n_class_correct.data.numpy() * 1.0 / n_total
#     acc_domain = n_domain_correct.data.numpy() * 1.0 / n_total
#     err_label = n_err_label.data.numpy() * 1.0 / len_dataloader
#     err_domain = n_err_domain.data.numpy() * 1.0 / len_dataloader
#
#     train_writer.add_scalar(tag="Loss/class", scalar_value=err_label, global_step=epoch)
#     train_writer.add_scalar(tag="Loss/domain", scalar_value=err_domain,global_step=epoch)
#
#     train_writer.add_scalar(tag="Acc/class", scalar_value=acc_class, global_step=epoch)
#     train_writer.add_scalar(tag="Acc/class", scalar_value=acc_class, global_step=epoch)
#     train_writer.add_scalar(tag="Acc/domain", scalar_value=acc_domain,global_step=epoch)
#     # train_writer.add_scalar(tag="loss_train/all", scalar_value=err, global_step=epoch * len_dataloader + i)
#     # train_writer.add_scalar(tag="params/lr", scalar_value=lr, global_step=epoch * len_dataloader + i)
#
#
#     # test_writer.add_scalar(tag="acc_val/source_class", scalar_value=accu_s_class, global_step=epoch)
#     # test_writer.add_scalar(tag="acc_val/source_domain", scalar_value=accu_s_domain, global_step=epoch)
#     test_writer.add_scalar(tag="Acc/class", scalar_value=accu_t_class, global_step=epoch )
#     test_writer.add_scalar(tag="Acc/domain", scalar_value=accu_t_domain, global_step=epoch)
#     test_writer.add_scalar(tag="Loss/domain", scalar_value=err_t_domain, global_step=epoch)
#     test_writer.add_scalar(tag="Loss/class", scalar_value=err_t_label, global_step=epoch)
#
#     if accu_t_class > best_accu_t:
#         best_accu_s = accu_s_class
#         best_accu_t = accu_t_class
#         torch.save(my_net, '{0}/{1}_{2}_model_epoch_best.pth'.format(model_root, source_dataset_name, target_dataset_name))
#
# print('============ Summary ============= \n')
# print('Accuracy of the %s dataset: %f' % (source_dataset_name, best_accu_s))
# print('Accuracy of the %s dataset: %f' % (target_dataset_name, best_accu_t))
# print('Corresponding model was save in {0}/{1}_{2}_model_epoch_best.pth'
#       .format(model_root, source_dataset_name, target_dataset_name))
