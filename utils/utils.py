import random
from torch.utils.data import Subset, random_split


def split_dataset(dataset, train_size=0.8):
    """
    :param dataset:
    :param train_size: 0 to 1
    :param random_seed:
    :return:
    """
    # random.seed(random_seed)

    # Compute the sizes of training and testing subsets
    train_len = int(train_size * len(dataset))
    test_len = len(dataset) - train_len

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    return train_dataset, test_dataset

def optimizer_scheduler(optimizer, lr,p, alpha=10, beta=0.75):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1. + alpha * p) ** beta
        lr = lr / (1. + alpha * p) ** beta

    return optimizer, lr