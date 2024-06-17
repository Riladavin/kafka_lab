from einops import rearrange
from collections import OrderedDict
from torchvision import models
from torch import nn
import numpy as np
from tqdm import tqdm
import torch


class img_fe_class(nn.Module):
    def __init__(self, model):
        super(img_fe_class, self).__init__()
        self.model = model

    def forward(self, imgs):
        return self.model.forward(imgs)


class text_fe_class(nn.Module):
    def __init__(
            self,
            num_layers,
            num_captions,
            img_features_size,
            hidden_size,
            dropout_p,
            vocab,
            glove_weights):
        super(text_fe_class, self).__init__()

        self.num_layers = num_layers
        self.num_captions = num_captions
        self.img_features_size = img_features_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embed = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=300,
            padding_idx=vocab['<PAD>'])
        self.embed.weight = nn.Parameter(
            torch.from_numpy(glove_weights).to(dtype=self.embed.weight.dtype),
            requires_grad=True,
        )

        self.adapt_img_features = nn.Linear(
            self.img_features_size, self.hidden_size)
        self.rnn = nn.GRU(
            input_size=300,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_p,
            bias=True,
            batch_first=True,
        )

        self.rnn.flatten_parameters()

    def forward(self, texts, img_features):
        emb = self.embed(texts)
        emb = rearrange(emb, "bs cap seq emb -> (bs cap) seq emb")

        img_features = self.adapt_img_features(img_features)
        img_features = img_features[None, :, None, :].repeat(
            self.num_layers, 1, self.num_captions, 1)
        img_features = rearrange(
            img_features,
            "lay bs cap hid -> lay (bs cap) hid")

        features, _ = self.rnn(emb, img_features)

        return rearrange(
            features,
            "(bs cap) seq hid -> bs cap seq hid",
            bs=texts.shape[0])


class image_captioning_model(nn.Module):
    def __init__(
            self,
            vocab_size,
            img_fe_params=None,
            text_fe_params=None,
            hid_size=256,
            dropout_p=0.3):
        super(image_captioning_model, self).__init__()

        self.img_fe_params = img_fe_params or {}
        self.text_fe_params = text_fe_params or {}
        self.hid_size = hid_size
        self.dropout_p = dropout_p
        self.vocab_size = vocab_size

        self.img_fe = img_fe_class(**self.img_fe_params)
        self.text_fe = text_fe_class(**self.text_fe_params)

        seq = [
            ('in2hid', nn.Linear(self.text_fe_params['hidden_size'], self.hid_size)),
            ('act', nn.ReLU()),
            ('norm', nn.LayerNorm(self.hid_size)),
            ('drop', nn.Dropout(self.dropout_p)),
            ('hid2out', nn.Linear(self.hid_size, self.vocab_size)),
        ]

        seq.append(('log_soft', nn.LogSoftmax(dim=-1)))

        self.fc = nn.Sequential(OrderedDict(seq))

    def forward(self, img_batch, texts_batch):
        feats_img = self.img_fe(img_batch)
        feats_text = self.text_fe(texts_batch, feats_img)

        return self.fc(feats_text)


np.random.seed(251200)


def load_glove_weights(file_path, vocab, pad_token="<PAD>"):
    print("Loading Glove Weights")
    glove_weights = np.random.uniform(0, 1, (len(vocab), 300))
    mask_found = np.zeros(len(vocab), dtype=bool)

    with open(file_path, 'r') as f:
        for line in tqdm(f, total=2196018):
            line = line.split()
            token = ' '.join(line[:-300])
            embed = line[-300:]

            if token in vocab:
                ind = vocab[token]
                mask_found[ind] = True
                glove_weights[ind, :] = np.array(
                    list(map(float, embed)), dtype=float)

    print(f"{mask_found.sum()} words from vocab of size {len(vocab)} loaded!")

    glove_weights[vocab[pad_token]] = np.zeros(300, dtype=float)
    return glove_weights, mask_found


def make_model_and_optimizer(data_preprocessor):
    glove_path = "./glove.840B.300d.txt"
    glove_weights, mask_found = load_glove_weights(
        glove_path, data_preprocessor.get_vocab())

    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights, progress=True)
    model = image_captioning_model(
        vocab_size=len(data_preprocessor.get_vocab()),
        img_fe_params=dict(
            model=resnet
        ),
        text_fe_params=dict(
            num_layers=2,
            num_captions=1,
            img_features_size=1000,
            hidden_size=256,
            dropout_p=0.1,
            vocab=data_preprocessor.get_vocab(),
            glove_weights=glove_weights,
        ),
    )

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr, [beta1, beta2])

    return model, optimizer
