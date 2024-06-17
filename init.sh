python3 -m venv ./venv
source ./venv/bin/activate

pip3 install -r requirements.txt

python3 download_data.py

python3 train_model.py