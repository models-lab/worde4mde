# word2vec-mde

Set-up:
```shell
sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader all
```

Download the dataset and put it in a folder:
```shell
TODO
```

Run the training procedure (by default all pdfs have to be in the folder docs):
```shell
python main.py --train
```

Test with it with word similarity:
```shell
python main.py --test
```