# Hierarchical Lexicon Embedding Architecture for CNER
This is the implementation of our paper "Hierarchical Lexicon Embedding Architecture for Chinese Named Entity Recognition", our method keeps high efficiency, high performance and high transferability at the same time, more details can be found at paper.

## Source code description
### Requirement:
Python 3.6  
Pytorch 0.4.1  
jieba  
### Input format:
CoNLL format, with each character and its label split by a whitespace in a line. The "BMES" tag scheme is prefered.

走 O  
过 O  
南 B-GPE  
京 M-GPE  
市 E-GPE  
长 B-LOC  
江 M-LOC  
大 M-LOC  
桥 E-LOC  
### Pretrain embedding:
The POS tag and relative position embeddings are randomly initialized, while the word embedding, char embedding and bichar embedding are the same with [Lattice LSTM](https://www.aclweb.org/anthology/P18-1144/)

### Run the code:
1. Download the character embeddings and word embeddings from [Lattice LSTM](https://www.aclweb.org/anthology/P18-1144/) and put them in the `data` folder.
2. Download your datasets and put them in the `data` folder.
3. To train on the dataset:
```
python main.py --model_type MODEL_TYPE --train TRAIN_FILE_PATH --dev DEV_FILE_PATH --test TEST_FILE_PATH --model_path MODEL_PATH --modelname MODEL_NAME --savedset DATASET_SAVE_PATH --num_iter NUM_ITER --seed SEED --hidden_dim HIDDEN_DIM --batch_size BATCH_SIZE --drop DROP --lr LR  
```
