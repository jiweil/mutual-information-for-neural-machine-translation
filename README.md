# Mutual Information and Diverse Decoding Improve Neural Machine Translation

Implementations of the three models presented in the paper "Mutual Information and Diverse Decoding Improve Neural Machine Translation" by Jiwei Li and Dan Jurafsky.

## Requirements:
GPU 

matlab >= 2014b

memory >= 8GB


## Folders
Standard: MMI reranking for standard sequence-to-sequence models

Standard/training: training p(t|s) and p(s|t)

Standard/decode: generating N-best list from p(t|s)

Standard/get_s_given_t: generating the score of p(s|t) 

Standard/MMI_rerank: reranking using different features including p(t|s) and p(s|t)

Attention: MMI reranking for attention models. 

data_gr: A sample of training/dev/testing data.

## Pipelines
(1) Training p(t|s) and p(s|t)

cd training

run matlab LSTM(1) or Attention(1) to train p(english|german)

run matlab LSTM(0) or Attention(1) to train p(german|english)

(2) generating N-best list from p(t|s)

cd decode 

run matlab decode()

(3) generating the backward score of p(s|t)

cd get_s_given_t

(3.a) prepare data

python generate_source_target.py 

(3.b) generating p(s|t)

matlab generate_score()

(d) feature reranking

cd MMI_rerank

Use the open package of MERT. If you don't have mert, you can do simple grid search by running

python tune_bleu.py. 

Monolingual features are not currently not included.


For any related questions, feel free to contact jiweil@stanford.edu

```latex
@article{li2016mutual,
  title={Mutual Information and Diverse Decoding Improve Neural Machine Translation},
  author={Li, Jiwei and Jurafsky, Dan},
  journal={arXiv preprint arXiv:1601.00372},
  year={2016}
}

```
