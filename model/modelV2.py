import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import ReverseLayerF


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output



class FeatureExtractorLeakly(nn.Module):
    def __init__(self):
        super(FeatureExtractorLeakly, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.bn2 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(x)
        x = self.pool(x)
        # x = F.leaky_relu(self.conv3(x))
        # x = self.pool(x)
        # return x.view(x.size(0), -1)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x = F.relu(self.conv3(x))
        # x = self.pool(x)
        return x.view(x.size(0), -1)


class FeatureExtractor2(nn.Module):
    def __init__(self):
        super(FeatureExtractor2, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.bn3(self.conv3(x))
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.bn4(self.conv4(x))
        x = F.leaky_relu(x)
        x = self.pool(x)
        # x = F.leaky_relu(self.conv3(x))
        # x = self.pool(x)
        # return x.view(x.size(0), -1)
        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(2816, 256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(256, 4)
        self.bn2 = nn.BatchNorm1d(num_features=4)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.bn1(self.fc1(x))
        x = F.leaky_relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.bn2(self.fc2(x))
        x = F.leaky_relu(x)
        x = self.softmax(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(2816, 256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(256, 2)
        self.bn2 = nn.BatchNorm1d(num_features=2)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.bn1(self.fc1(x))
        x = F.leaky_relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.bn2(self.fc2(x))
        x = F.leaky_relu(x)
        x = self.softmax(x)
        return x


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor2()
        self.label_predictor = LabelPredictor()
        self.domain_classifier = DomainClassifier()

    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        reverse_feature = ReverseLayerF.apply(features, alpha)
        label_preds = self.label_predictor(features)
        domain_preds = self.domain_classifier(reverse_feature)
        return label_preds, domain_preds


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.label_predictor = LabelPredictor()

    def forward(self, x):
        features = self.feature_extractor(x)
        label_preds = self.label_predictor(features)
        return label_preds

