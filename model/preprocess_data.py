from torchvision import transforms as tr
import numpy as np
from sklearn.preprocessing import minmax_scale
from tqdm.auto import trange
from collections import Counter
import re
import random
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch


def tokenize(text):
    text = re.sub(r'[^\w\s]+', ' ', text.lower())
    return ['<BOS>'] + text.split() + ['<EOS>']


def de_normalize(img, channel_mean, channel_std):
    return minmax_scale(
        (img.reshape(3, -1) + channel_mean[:, None]) * channel_std[:, None],
        feature_range=(0., 1.),
        axis=1,
    ).reshape(*img.shape)


class DataPreprocessor:
    global_max_seq_len = 0

    def __init__(self, dfs, data_folder='./data'):
        filenames = dfs['train']['img_id'].values.tolist()

        self.channel_mean = 0
        count = 0

        for file in filenames:
            image = cv2.imread(os.path.join(data_folder, 'train', file)) / 255
            image = cv2.resize(image, (256, 256))
            self.channel_mean += np.sum(image, axis=(0, 1))
            count += image.shape[0] * image.shape[1]
        self.channel_mean /= count

        self.channel_std = 0
        for file in filenames:
            img = cv2.imread(os.path.join(data_folder, 'train', file)) / 255
            img = cv2.resize(img, (256, 256))
            img -= self.channel_mean
            self.channel_std += np.sum(np.square(img), axis=(0, 1))
        self.channel_std /= count
        self.channel_std = np.sqrt(self.channel_std)

        self.image_prepare = tr.Compose([
            tr.ToPILImage(),
            tr.ToTensor(),
            tr.Normalize(mean=self.channel_mean, std=self.channel_std),
        ])

        vocab_freq = Counter()
        sizes = Counter()
        for i in trange(len(dfs['train'])):
            row = dfs['train'].iloc[i]
            for j in range(5):
                tokens = tokenize(row[f'caption #{j}'])[1:-1]
                vocab_freq.update(tokens)
                sizes[len(tokens)] += 1
                DataPreprocessor.global_max_seq_len = max(
                    DataPreprocessor.global_max_seq_len, len(tokens))

        MIN_FREQ = 3

        self.tok_to_ind = {
            '<UNK>': 0,
            '<BOS>': 1,
            '<EOS>': 2,
            '<PAD>': 3,
        }

        self.ind_to_tok = {
            0: '<UNK>',
            1: '<BOS>',
            2: '<EOS>',
            3: '<PAD>',
        }

        for key in vocab_freq.keys():
            if vocab_freq[key] >= MIN_FREQ:
                id = len(self.ind_to_tok)
                self.ind_to_tok[id] = key
                self.tok_to_ind[key] = id

        self.vocab_size = len(self.tok_to_ind)

    def to_ids(self, text):
        return [self.tok_to_ind[token]
                if token in self.tok_to_ind else 0 for token in tokenize(text)]

    def process_image(self, image):
        return self.image_prepare(image)

    def get_vocab(self):
        return self.tok_to_ind

    def get_ind_to_tok(self):
        return self.ind_to_tok


class ImageCaptioningDataset(Dataset):
    """
        imgs_path ~ путь к папке с изображениями
        captions_path ~ путь к .tsv файлу с заголовками изображений
    """

    def __init__(self, imgs_path, df, data_preprocessor):
        self.processor = data_preprocessor

        super(ImageCaptioningDataset).__init__()
        self.images = []
        self.captions = []
        for i in range(len(df)):
            row = df.iloc[i]
            image = cv2.imread(os.path.join(imgs_path, row['img_id'])) / 255
            image = cv2.resize(image, (256, 256))
            image = self.processor.image_prepare(image)
            self.images.append(image)
            self.captions.append([self.processor.to_ids(
                row[f'caption #{j}']) for j in range(5)])

    def __getitem__(self, index):
        return self.images[index], self.captions[index][random.randrange(5)]

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    images = []
    captions = []
    tensor_shape = (1,) + batch[0][0].shape
    for i in range(len(batch)):
        images.append(batch[i][0].reshape(tensor_shape))
        caption_tensor = torch.LongTensor(
            batch[i][1] + [0] * (DataPreprocessor.global_max_seq_len - len(batch[i][1])))
        caption_tensor = caption_tensor.reshape((1, 1,) + caption_tensor.shape)
        if caption_tensor.shape[-1] > DataPreprocessor.global_max_seq_len:
            caption_tensor = caption_tensor[:, :,
                                            :DataPreprocessor.global_max_seq_len]
        captions.append(caption_tensor)

    img_batch = torch.concat(images)
    captions_batch = torch.concat(captions)

    return img_batch, captions_batch


def make_dataloader(dataset, train=True, batch_size=64):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=train,
        drop_last=train,
        num_workers=0,
    )
