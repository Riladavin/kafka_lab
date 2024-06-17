import json

import torch
from confluent_kafka import Consumer, Producer
from model.preprocess_data import DataPreprocessor
from model.dataset_loader import load_dataset
from typing import Optional
import numpy as np
import cv2
import os

from config import (data_processor_config,
                    data_processor_topic_config,
                    data_producer_config,
                    data_producer_topic_config)

def generate(
        model,
        preprocessor,
        image,
        max_seq_len,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        img = preprocessor.process_image(image)[None, :, :, :].to(device)
        tok_to_ind = preprocessor.get_vocab()
        text = [tok_to_ind['<BOS>']]
        generated_size = 0
        while text[generated_size] != tok_to_ind['<EOS>'] and generated_size + 1 < max_seq_len:
            caption = torch.tensor(text, dtype=torch.int64)[None, None, :].repeat(1, 1, 1).to(device)
            pred = model(img, caption)
            probs = pred.detach().cpu().exp().numpy()[0, 0, -1, :]

            if top_k is not None:
                inds = np.argsort(-probs)[:top_k]
                pred_ind = np.random.choice(inds, p=probs[inds] / probs[inds].sum())
            elif top_p is not None:
                inds = np.argsort(-probs)
                while probs[inds].sum() > top_p and len(inds) > 1:
                    inds = inds[:-1]
                pred_ind = np.random.choice(inds, p=probs[inds] / probs[inds].sum())
            else:
                pred_ind = np.random.choice(np.arange(probs.shape[0]), p=probs)
            text.append(pred_ind)
            generated_size += 1

        if text[generated_size] == tok_to_ind['<EOS>']:
            result_tokens = [preprocessor.get_ind_to_tok()[ind] for ind in text[1:-1]]
            result_text = ' '.join(result_tokens)
        else:
            result_tokens = [preprocessor.get_ind_to_tok()[ind] for ind in text[1:]]
            result_text = ' '.join(result_tokens + ['...'])

        return result_tokens, result_text


def main():
    topic_consume = [data_producer_topic_config]
    conf_consume = {**data_producer_config, 'group.id': 'data_processors'}
    consumer = Consumer(conf_consume)
    consumer.subscribe(topic_consume)

    producer = Producer(data_processor_config)

    checkpoint_path = 'data/checkpoints/best'

    model = torch.load(checkpoint_path)
    model.eval()

    dfs = load_dataset()

    preprocessor = DataPreprocessor(dfs)

    print(DataPreprocessor.global_max_seq_len)

    while True:
        msg = consumer.poll(timeout=1000)

        if msg is not None:
            data = json.loads(msg.value().decode('utf-8'))

            img_id = data['img_id']
            img = cv2.imread(os.path.join('./data/val', img_id))

            caption_tokens, caption_text = generate(model, preprocessor, img, DataPreprocessor.global_max_seq_len, top_k=2)

            result_data = {
                'caption_text': caption_text,
                'caption_tokens': caption_tokens,
                'reference_captions': [dfs['val'].iloc[data['sample_id']][f'caption #{i}'] for i in range(5)]
            }

            producer.produce(
                data_processor_topic_config,
                key='1',
                value=json.dumps(result_data))
            producer.flush()

            sample_id = data['sample_id']
            print(f'Processed: {sample_id}', flush=True)


if __name__ == '__main__':
    main()
