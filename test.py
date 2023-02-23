import os
import torch.backends.cudnn as cudnn
import torch.utils.data

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

def test(dataloader, is_target=True, model_root=''):


    # model_root = 'models_best_0216'

    cuda = True
    cudnn.benchmark = True
    alpha = 0


    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_class_correct = 0
    n_domain_correct = 0
    n_err_label = 0
    n_err_domain = 0


    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.__next__()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, domain_output = my_net(x=t_img, alpha=alpha)

        if is_target:
            domain_label = torch.ones(batch_size).long()
        else:
            domain_label = torch.zeros(batch_size).long()

        if cuda:
            domain_label = domain_label.cuda()


        pred_class = class_output.data.max(1, keepdim=True)[1]
        n_class_correct += pred_class.eq(t_label.data.view_as(pred_class)).cpu().sum()
        pred_domain = domain_output.data.max(1, keepdim=True)[1]
        n_domain_correct += pred_domain.eq(domain_label.data.view_as(pred_domain)).cpu().sum()

        n_total += batch_size

        n_err_label += loss_class(class_output, t_label.long()).cpu()
        n_err_domain += loss_domain(domain_output, domain_label).cpu()

        i += 1

    acc_class = n_class_correct.data.numpy() * 1.0 / n_total
    acc_domain = n_domain_correct.data.numpy() * 1.0 / n_total


    err_label = n_err_label.data.numpy() * 1.0 / len_dataloader
    err_domain = n_err_domain.data.numpy() * 1.0 / len_dataloader

    return acc_class, acc_domain, err_label, err_domain
