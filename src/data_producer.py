import json
import random
import time

from model.dataset_loader import load_dataset
from confluent_kafka import Producer

from config import data_producer_config, data_producer_topic_config

def main():
    producer1 = Producer(data_producer_config)
    producer2 = Producer(data_producer_config)

    dataset = load_dataset()

    while True:
        sample_1_id = random.randint(0, len(dataset['val']) - 1)
        sample_2_id = random.randint(0, len(dataset['val']) - 1)

        sample_1 = dataset['val'].iloc[sample_1_id]
        sample_2 = dataset['val'].iloc[sample_2_id]

        sample_1_dict = {
            'img_id': sample_1['img_id'],
            'sample_id': sample_1_id,
        }

        sample_2_dict = {
            'img_id': sample_2['img_id'],
            'sample_id': sample_2_id,
        }

        producer1.produce(
            data_producer_topic_config,
            key='1',
            value=json.dumps(sample_1_dict))
        producer2.produce(
            data_producer_topic_config,
            key='1',
            value=json.dumps(sample_2_dict))

        producer1.flush()
        producer2.flush()

        print(f'Produced sample: {sample_1_id}')
        print(f'Produced sample: {sample_2_id}')
        time.sleep(10 + random.uniform(0, 5.0))


if __name__ == '__main__':
    main()
