# KGIRNET: Grounding DIalogue Systems via Knowledge Graph Aware Decoding with Pre-trained Transformers


### Requirements
* python==3.6
* torch==1.3.1


### Installation
First download the Fasttext word embedding from here ([download](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip)), extract and put the wiki.simple.bin file inside ```data/``` directory.


Now run the following command to install the requirements:
```python
pip install -r requirements.txt
```

To test the already trained model download the saved model form here ([download](https://gofile.io/d/2v3Kyo)) and put it inside ```saved_models/``` directory. Now run the following command:
```python
python -u ./train_kgirnet.py --batch_size 20 --hidden_size 256 --rnn_dropout 0.2 --dropout 0.3 --decoder_lr 10 --epochs 10 --teacher_forcing 10 --resp_len 20 --lr 0.0001 --use_bert 1 --dataset incar --evaluate 1
```


To train the model from scratch run the following command:
```python
python -u ./train_kgirnet.py --batch_size 20 --hidden_size 256 --rnn_dropout 0.2 --dropout 0.3 --decoder_lr 10 --epochs 10 --teacher_forcing 10 --resp_len 20 --lr 0.0001 --use_bert 1 --dataset incar
```

### Evaluation
To get the METEOR score of the saved predictions run:
```python
python evaluators/METEOR_score.py
```