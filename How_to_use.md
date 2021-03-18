# ref: https://spacy.io/usage/models


# 1) Download language models:
# English:
python -m spacy download en_core_web_sm
# French:
python -m spacy download fr_core_news_sm


# 2) Training:
python train.py -batchsize 6000 -epochs 1 -output_dir output_epochs_1

# 3) Test:
python translate.py -load_weights output_epochs_1
