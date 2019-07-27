import os

import requests

release_root = 'https://github.com/applecrazy/reportik/releases/latest/download/'
files = [
    'cnbc_tokenizers.pkl',
    'test_cnbc_data.pkl',
    'train_cnbc_data.pkl',
    'doc_emb.bin',
    'summ_emb.bin'
]

data_dir = os.path.join(os.getcwd(), 'data')

# Check if the data directory exists, if not create it
if not os.path.isdir(data_dir):
    print('[INFO] Creating', data_dir)
    os.mkdir(data_dir)

for file in files:
    temp_path = os.path.join(data_dir, file)
    print('[INFO] Downloading and saving', file, 'at', temp_path)
    with open(temp_path, 'wb+') as f:
        res = requests.get(release_root + file)
        f.write(res.content)

print('[INFO] Done.')
