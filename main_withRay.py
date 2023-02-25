import os
import sys
import torch
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils.data_loader import get_dataset
from model.modelV2 import DANN
from utils.utils import split_dataset
# from test import test
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# add summeryWriter
# train_writer = SummaryWriter(log_dir=model_root + "tb/train")
# test_writer = SummaryWriter(log_dir=model_root + "tb/test")

# cuda = True
# cudnn.benchmark = True
# lr = 1e-3
# theta = 1
# batch_size = 64
# n_epoch = 200
#
# manual_seed = random.randint(1, 10000)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
#
# # load model
# my_net = DANN()
#
#
# # setup optimizer
# optimizer = optim.Adam(my_net.parameters(), lr=lr)

# load data
def load_data(dataset_name, dataset_root='dataset/8positions_200/features_shuffle/',train_size=0.8):
    path = os.path.join(dataset_root, dataset_name)
    data_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.mat')]
    train_dataset, test_dataset = get_dataset(data_files,train_size=train_size)
    return train_dataset, test_dataset

def train_dann(config, ckp_dir=None, data_dir=None):
    # load config
    batch_size = config['batch_size']
    lr = config['lr']


    n_epoch = 100
    theta = 1

    # load model
    my_net = DANN()

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    # setting optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    cuda = True
    cudnn.benchmark = True
    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # make dataLoader
    source_dataset_name = 'sunny'
    target_dataset_name = 'snowy'
    dataset_source_train, dataset_source_test = load_data(dataset_name=source_dataset_name)
    dataloader_source_train = DataLoader(dataset_source_train, batch_size=batch_size, shuffle=True)
    dataloader_source_test = DataLoader(dataset_source_test, batch_size=batch_size, shuffle=True)
    dataset_target_train, dataset_target_test = load_data(dataset_name=target_dataset_name)
    dataloader_target_train = DataLoader(dataset_target_train, batch_size=batch_size, shuffle=True)
    dataloader_target_test = DataLoader(dataset_target_test, batch_size=batch_size, shuffle=True)

    # 用于存储检查点
    if ckp_dir:
        # 模型的状态、优化器的状态
        model_state, optimizer_state = torch.load(
            os.path.join(ckp_dir, "checkpoint"))
        my_net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)



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
            # alpha = 0.05

            # training model using source data
            data_source = data_source_iter.__next__()
            s_img, s_label = data_source

            # optimizer, lr_current = optimizer_scheduler(optimizer=optimizer, lr=lr,p=p)
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
            # pred_class = class_output.data.max(1, keepdim=True)[1]
            # n_class_correct += pred_class.eq(s_label.data.view_as(pred_class)).cpu().sum()
            # pred_domain = domain_output.data.max(1, keepdim=True)[1]
            # n_domain_correct += pred_domain.eq(domain_label.data.view_as(pred_domain)).cpu().sum()
            # n_total += batch_size
            # n_err_label += loss_class(class_output, s_label.long()).cpu()
            # n_err_domain += loss_domain(domain_output, domain_label).cpu()

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

            # train_writer.add_scalar(tag="loss_training/all", scalar_value=err, global_step=epoch * len_dataloader + i)
            # train_writer.add_scalar(tag="loss_training/domain", scalar_value=err_t_domain + err_s_domain,
            #                         global_step=epoch * len_dataloader + i)
            # train_writer.add_scalar(tag="loss_training/class", scalar_value=err_s_label,
            #                         global_step=epoch * len_dataloader + i)
            # train_writer.add_scalar(tag="params/alpha", scalar_value=alpha, global_step=epoch * len_dataloader + i)
            # train_writer.add_scalar(tag="params/lr_current", scalar_value=lr_current, global_step=epoch * len_dataloader + i)

        #################### training summary in one epoch ############################
        val_loss_class = 0.0
        val_loss_domain = 0.0
        val_steps = 0
        total = 0
        correct_class = 0
        correct_domain = 0
        for i, data in enumerate(dataloader_target_test, 0):
            with torch.no_grad():
                t_img, t_label = data
                t_img, t_label = t_img.cuda(), t_label.cuda()

                class_output, domain_output = my_net(x=t_img,alpha=0)

                domain_label = torch.ones(batch_size).long()


                _, predicted = torch.max(class_output.data, 1)
                total += t_label.size(0)
                correct_class += (predicted == t_label).sum().item()

                _, predicted = torch.max(domain_output.data, 1)
                total += domain_label.size(0)
                correct_domain += (predicted == domain_label).sum().item()

                class_loss = loss_class(class_output, t_label)
                val_loss_class += class_loss.cpu().numpy()

                class_loss = loss_class(class_output, t_label)
                val_loss_class += class_loss.cpu().numpy()

                val_steps += 1


        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((my_net.state_dict(), optimizer.state_dict()), path)
            # 打印平均损失和平均精度
        tune.report(loss_class=(val_loss_class / val_steps), accuracy_class=correct_class / total,
                    loss_domain=(val_loss_domain / val_steps), accuracy_domain=correct_domain / total)
    print("Finished Training")


# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
#     # 全局文件路径
#     data_dir = os.path.abspath("/home/taoshouzheng/Local_Connection/Algorithms/ray/")
#
#     # 加载训练数据
#     # load_data(data_dir)
#
#     # 配置超参数搜索空间
#     # 每次实验，Ray Tune会随机采样超参数组合，并行训练模型，找到最优参数组合
#     config = {
#         # 自定义采样方法
#         # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         # 随机分布采样
#         "lr": tune.loguniform(1e-4, 1e-1),
#         # 从类别型值中随机选择
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#     # ASHAScheduler会根据指定标准提前中止坏实验
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2)
#     # 在命令行打印实验报告
#     reporter = CLIReporter(
#         # parameter_columns=["l1", "l2", "lr", "batch_size"],
#         metric_columns=["loss", "accuracy", "training_iteration"])
#     # 执行训练过程
#     result = tune.run(
#         train_dann,
#         name="exp_ray"
#         # 指定训练资源
#         resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         # scheduler=scheduler,
#         progress_reporter=reporter)
#
#     # 找出最佳实验
#     best_trial = result.get_best_trial("loss", "min", "last")
#     # 打印最佳实验的参数配置
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(
#         best_trial.last_result["loss"]))
#     print("Best trial final validation accuracy: {}".format(
#         best_trial.last_result["accuracy"]))
#
#     # 打印最优超参数组合对应的模型在测试集上的性能
#     # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
#     # device = "cpu"
#     # if torch.cuda.is_available():
#     #     device = "cuda:0"
#     #     if gpus_per_trial > 1:
#     #         best_trained_model = nn.DataParallel(best_trained_model)
#     # best_trained_model.to(device)
#     #
#     # best_checkpoint_dir = best_trial.checkpoint.value
#     # model_state, optimizer_state = torch.load(os.path.join(
#     #     best_checkpoint_dir, "checkpoint"))
#     # best_trained_model.load_state_dict(model_state)
#     #
#     # test_acc = test_accuracy(best_trained_model, device)
#     # print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)

    # # 全局文件路径
    # data_dir = os.path.abspath("")

    # 加载训练数据
    # load_data(data_dir)

    # 配置超参数搜索空间
    # 每次实验，Ray Tune会随机采样超参数组合，并行训练模型，找到最优参数组合
    config = {
        # 自定义采样方法
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # 随机分布采样
        "lr": tune.loguniform(1e-5, 1e-3),
        # 从类别型值中随机选择
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    # ASHAScheduler会根据指定标准提前中止坏实验
    # scheduler = ASHAScheduler(
    #     metric="loss",
    #     mode="min",
    #     max_t=max_num_epochs,
    #     grace_period=1,
    #     reduction_factor=2)
    # 在命令行打印实验报告
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss_class", "accuracy_class", "lr"])
    # 执行训练过程
    result = tune.run(
        train_dann,
        name="exp_ray",
        # 指定训练资源
        resources_per_trial = {"cpu": 8, "gpu": 2},
        config = config,
        num_samples = 20,
        # scheduler=scheduler,
        progress_reporter = reporter
    )

    print("======================== Result =========================")
    print(result.results_df)


    # # 找出最佳实验
    # best_trial = result.get_best_trial("loss", "min", "last")
    # # 打印最佳实验的参数配置
    # print("Best trial config: {}".format(best_trial.config))
    # print("Best trial final validation loss: {}".format(
    #     best_trial.last_result["loss"]))
    # print("Best trial final validation accuracy: {}".format(
    #     best_trial.last_result["accuracy"]))

    # 打印最优超参数组合对应的模型在测试集上的性能
    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)
    #
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)
    #
    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))



