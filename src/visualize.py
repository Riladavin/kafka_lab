import json
from collections import defaultdict

import matplotlib.pyplot as plt
from model.preprocess_data import DataPreprocessor
import streamlit as st
from confluent_kafka import Consumer
from collections import Counter
import pandas as pd

from config import (data_processor_config,
                    data_processor_topic_config)


def main():
    topic_consume = [data_processor_topic_config]
    conf_consume = {**data_processor_config, 'group.id': 'data_visualizers'}
    consumer = Consumer(conf_consume)
    consumer.subscribe(topic_consume)

    st.set_page_config(
        page_title='Real-Time Data Dashboard',
        layout='wide',
    )

    container_popular_words = st.container(border=True)
    container_popular_words.title('Most popular words')
    container_popular_words = container_popular_words.empty()

    container_lengths_of_generation = st.container(border=True)
    container_lengths_of_generation.title('Distribution of sequence lengths')
    container_lengths_of_generation = container_lengths_of_generation.empty()

    counter = Counter()
    length_counter = Counter()
    max_generation_len = 0

    while True:
        msg = consumer.poll(timeout=1000)

        if msg is not None:
            container_popular_words = container_popular_words.empty()
            container_lengths_of_generation = container_lengths_of_generation.empty()
            data = json.loads(msg.value().decode('utf-8'))

            tokens = data['caption_tokens']

            length_counter[len(tokens)] += 1
            max_generation_len = max(max_generation_len, len(tokens))

            counter.update(tokens)

            most_frequent = counter.most_common(5)

            text = data['caption_text']
            fig1, ax1 = plt.subplots()
            fig1.set_size_inches(4, 4)
            plt.title('most popular words')
            plt.bar([item[0] for item in most_frequent], [item[1] for item in most_frequent])

            ax1.set(
                ylabel='Count',
                title='Per word'
            )
            container_popular_words.pyplot(fig1, use_container_width=False)

            fig2, ax2 = plt.subplots()
            fig2.set_size_inches(4, 4)
            plt.title('Most popular words')
            plt.bar([i for i in range(1, max_generation_len + 1)],
                    [length_counter[i] for i in range(1, max_generation_len + 1)])

            ax2.set(
                ylabel='Count of generations',
                title='Sequence length'
            )
            container_lengths_of_generation.pyplot(fig2, use_container_width=False)
            plt.close()


if __name__ == '__main__':
    main()
