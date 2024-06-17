import requests
from urllib.parse import urlencode

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/2-ZV5OsM2H7Lkg'

final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

download_response = requests.get(download_url)
with open('dataset.tar.gz', 'wb') as f:
    f.write(download_response.content)

import os

os.system('tar xfz dataset.tar.gz -C .')
os.system('rm dataset.tar.gz')

os.system('wget -O glove.zip https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip')
os.system('unzip glove.zip')
