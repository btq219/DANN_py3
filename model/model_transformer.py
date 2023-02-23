import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import ReverseLayerF


class FeatureExtractor(nn.Module):
    def __init__(self,input_dim, num_heads=4, num_layers=2, hidden_dim=256, dropout=0.2):
        super(FeatureExtractor, self).__init__()

        # embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_embedding = nn.Embedding(179, hidden_dim)

        # transformer layer
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # classification layer
        # self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # positional embedding
        x_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x_pos = self.positional_embedding(x_pos)

        # add positional embedding to input
        x = self.embedding(x) + x_pos

        # transformer encoding
        x = x.transpose(0, 1)  # (B, S, E) -> (S, B, E)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # take only the last output of the transformer

        # classification
        # x = self.classifier(x)
        # x = F.log_softmax(x, dim=1)

        return x


class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.fc1 = nn.Linear(256, 256)   #5632
        self.fc2 = nn.Linear(256, 4)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.softmax(x)
        return x



class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim=179)
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
        self.feature_extractor = FeatureExtractor(input_dim=179)
        self.label_predictor = LabelPredictor()

    def forward(self, x):
        features = self.feature_extractor(x)
        label_preds = self.label_predictor(features)
        return label_preds