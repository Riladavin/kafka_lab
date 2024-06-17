import os
from model.dataset_loader import load_dataset
from model.preprocess_data import DataPreprocessor, ImageCaptioningDataset, make_dataloader
from model.model import make_model_and_optimizer
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from einops import rearrange

def main():
    checkpoint_path = './data/checkpoints'

    print('loading dataset')
    dfs = load_dataset()

    print('preprocessing dataset')
    preprocessor = DataPreprocessor(dfs)

    train_dataset = ImageCaptioningDataset('./data/train', dfs['train'], preprocessor)

    train_dataloader = make_dataloader(train_dataset)

    model, optimizer = make_model_and_optimizer(preprocessor)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                           threshold=0.001, verbose=True)

    criterion = nn.NLLLoss()
    model.train()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('start training model')

    for epoch in range(1, 2):
        losses_tr = []
        for img_batch, captions_batch in tqdm(train_dataloader):
            img_batch = img_batch.to(device)
            captions_batch = captions_batch.to(device)

            optimizer.zero_grad()

            pred = model(img_batch, captions_batch[:, :, :-1])

            pred = rearrange(pred, "bs cap seq voc -> (bs cap seq) voc")

            target = captions_batch[:, :, 1:].reshape(-1)

            loss = criterion(pred, target)

            loss.backward()
            optimizer.step()
            losses_tr.append(loss.item())
        print(f'epoch = {epoch}, loss={np.mean(losses_tr)}')
        torch.save(model, os.path.join(checkpoint_path, 'best'))

if __name__ == "__main__":
    main()
