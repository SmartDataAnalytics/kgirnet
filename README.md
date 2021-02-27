# KGIRNET: Grounding Dialogue Systems via Knowledge Graph Aware Decoding with Pre-trained Transformers
Generating knowledge grounded responses in both goal and non-goal oriented dialogue systems is an important research challenge. Knowledge Graphs (KG) can be viewed as an abstraction of the real world, which can potentially facilitate a dialogue system to produce knowledge grounded responses. However, integrating KGs into the dialogue generation process in an end-to-end manner is a non-trivial task. This paper proposes a novel architecture for integrating KGs into the response generation process by training a BERT model that learns to answer using the elements of the KG (entities and relations) in a multi-task, end-to-end setting. The k-hop subgraph of the KG is incorporated into the model during training and inference using Graph Laplacian. Empirical evaluation suggests that the model achieves better knowledge groundedness (measure via entity F1 score) compared to other state-of-the-art models for both goal and non-goal oriented dialogues.
![](https://github.com/DeepInEvil/kgirnet/blob/main/model_diagram.png)

### Requirements
* python==3.6
* torch==1.5.1
* [Anaconda](https://www.anaconda.com/products/individual)


### Installation
First download the Fasttext word embedding from here ([download](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip)), extract and put the wiki.simple.bin file inside ```data/``` directory.


Now run the following command to install the requirements:
```commandline
pip install -r requirements.txt
```
## Test
To test the pre-trained model download the saved model form here ([in-car model](https://gofile.io/d/2v3Kyo), [soccer model](https://gofile.io/d/ZqoNwM)) and put them inside the ```saved_models/``` directory. Now run the following commands:
#### For in-car dataset:
```python
python -u ./train_kgirnet.py --batch_size 20 --hidden_size 256 --rnn_dropout 0.2 --dropout 0.3 --decoder_lr 10 --epochs 10 --teacher_forcing 10 --resp_len 20 --lr 0.0001 --use_bert 1 --dataset incar --evaluate 1
```
#### For soccer dataset:
```python
python -u ./train_kgirnet.py --batch_size 20 --hidden_size 256 --rnn_dropout 0.2 --dropout 0.3 --decoder_lr 10 --epochs 10 --teacher_forcing 10 --resp_len 20 --lr 0.0001 --use_bert 1 --dataset soccer --evaluate 1
```

## Train
To train the model from scratch run the following command:
#### For in-car dataset
```python
python -u ./train_kgirnet.py --batch_size 20 --hidden_size 256 --rnn_dropout 0.2 --dropout 0.3 --decoder_lr 10 --epochs 10 --teacher_forcing 10 --resp_len 20 --lr 0.0001 --use_bert 1 --dataset incar
```
#### For soccer dataset
```python
python -u ./train_kgirnet.py --batch_size 20 --hidden_size 256 --rnn_dropout 0.2 --dropout 0.3 --decoder_lr 10 --epochs 10 --teacher_forcing 10 --resp_len 20 --lr 0.0001 --use_bert 1 --dataset soccer
```

## Evaluation
To get the METEOR score of the saved predictions run:
```python
python evaluators/METEOR_score.py
```
