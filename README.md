# CG-nAR

Pytorch implementation of the EMNLP-2021 paper: [Thinking Clearly, Talking Fast: Concept-Guided Non-Autoregressive Generation for Open-Domain Dialogue Systems](https://arxiv.org/abs/2109.04084).

## Requirements

* python 3.7.10

* torch 1.7.0+cu11.0

* transformers 4.0.0

* multiprocess 0.70.11.1

* tensorboardX 2.1

* torchtext 0.8.0

* nltk 3.6.2 

    nltk packages: wordnet_ic, averaged_perceptron_tagger, wordnet

* tensorflow 1.12

* scipy 1.7.1

* matplotlib 3.4.3

* sklearn 1.0

* munkres 1.1.4

* rouge 1.0.0

## Environment

* RTX 3090 GPU

* CUDA version 11.1

## Data

* `Persona-Chat`:

The way to divide the persona dataset comes from https://github.com/squareRoot3/Target-Guided-Conversation. You can get the raw dataset at [google drive](https://drive.google.com/file/d/1oTjOQjm7iiUitOPLCmlkXOCbEPoSWDPX/view?usp=sharing). Download and unzip it into the directory `resource`. 

Besides, we use `glove.6B.300d` to initialize the graph-embedding. You can download it from [fastnlp](http://download.fastnlp.top/embedding/glove.6B.300d.zip). Download and unzip it into the directory `resource`.

Finally, run the script for data preprocessing:
```shell
bash src/persona_preprocess.sh
```
By default, the divided dataset will be put into the `raw_data/persona` directory and the graph-related data will be put into the `graph_data/persona` directory.

* `Weibo`:

You can download the cleaned dataset at [Baidu Pan(extract code: kbiz)](https://pan.baidu.com/s/1awh3_ojU2AN4V536UINp0w). Download and unzip it into the directory `resource`. 

We use `cn_bi_fastnlp_100d` to initialize the graph-embedding. You can download it from [fastnlp](http://download.fastnlp.top/embedding/cn_bi_fastnlp_100d.zip). Download and unzip it into the directory `resource`. 

After that, run the script for data preprocessing to get the necessary data in the program:
```shell
bash src/weibo_preprocess.sh
```
By default, the dataset will be put into the `raw_data/weibo` directory and the graph-related data will be put into the `graph_data/weibo` directory.

## Usage

* `Persona-Chat`

1. Pre-process

```
PYTHONPATH=. python ./src/preprocess.py -dataset persona -mode raw_to_json -raw_path raw_data/persona -save_path json_data/persona/persona -adj_file graph_data/persona/adj_matrix.txt -vertex_file graph_data/persona/vertex.txt -log_file logs/raw_to_json_persona.log
```

```
PYTHONPATH=. python ./src/preprocess.py -dataset persona -mode json_to_data -type train -raw_path json_data/persona -save_path torch_data/persona -tokenizer bert-base-uncased -adj_file graph_data/persona/adj_matrix.txt -vertex_file graph_data/persona/vertex.txt -log_file logs/json_to_data_persona.log -n_cpus 4
```

2. Train

```
PYTHONPATH=. python ./src/main.py -mode train -data_path torch_data/persona/persona -model_path models/persona -log_file logs/persona.train.log -visible_gpus 0 -warmup_steps 8000 -lr 0.001 -train_steps 100000 -graph_emb_path graph_data/persona/graph_embedding.npy -tokenizer bert-base-uncased
```

3. Validate

```
PYTHONPATH=. python ./src/main.py -mode validate -data_path torch_data/persona/persona -log_file logs/persona.val.log -test_all -alpha 0.95 -model_path models/persona -result_path results/persona/persona -test_start_from 10000 -visible_gpus 0 -test_batch_ex_size 50 -graph_emb_path graph_data/persona/graph_embedding.npy -tokenizer bert-base-uncased
```

4. Test
```
PYTHONPATH=. python ./src/main.py -mode test -data_path torch_data/persona/persona -log_file logs/persona.test.log -alpha 0.95 -test_from models/persona/model_step_100000.pt -result_path results/persona/persona -visible_gpus 0 -test_batch_ex_size 50 -graph_emb_path graph_data/persona/graph_embedding.npy -tokenizer bert-base-uncased
```

* `Weibo`
1. Pre-process
```
PYTHONPATH=. python ./src/preprocess.py -dataset weibo -mode raw_to_json -raw_path raw_data/weibo -save_path json_data/weibo/weibo -adj_file graph_data/weibo/adj_matrix.txt -vertex_file graph_data/weibo/vertex.txt -log_file logs/raw_to_json_weibo.log
```

```
PYTHONPATH=. python ./src/preprocess.py -dataset weibo -mode json_to_data -type train -raw_path json_data/weibo -save_path torch_data/weibo -tokenizer bert-base-chinese -adj_file graph_data/weibo/adj_matrix.txt -vertex_file graph_data/weibo/vertex.txt -log_file logs/json_to_data_weibo.log -n_cpus 8
```

2. Train
```
PYTHONPATH=. python ./src/main.py -mode train -data_path torch_data/weibo/weibo -model_path models/weibo -log_file logs/weibo.train.log -visible_gpus 0 -warmup_steps 8000 -lr 0.001 -train_steps 100000 -graph_emb_path graph_data/weibo/graph_embedding.npy -tokenizer bert-base-chinese
```

3. Validate
```
PYTHONPATH=. python ./src/main.py -mode validate -data_path torch_data/weibo/weibo -log_file logs/weibo.val.log -test_all -alpha 0.95 -model_path models/weibo -result_path results/weibo/weibo -test_start_from 10000 -visible_gpus 0 -test_batch_ex_size 50 -graph_emb_path graph_data/weibo/graph_embedding.npy -tokenizer bert-base-chinese
```

4. Test
```
PYTHONPATH=. python ./src/main.py -mode test -data_path torch_data/weibo/weibo -log_file logs/weibo.test.log -alpha 0.95 -test_from models/weibo/model_step_100000.pt -result_path results/weibo/weibo -visible_gpus 0 -test_batch_ex_size 50 -graph_emb_path graph_data/weibo/graph_embedding.npy -tokenizer bert-base-chinese
```

## Citation

    @inproceedings{
        zou-etal-2021-thinking,
        title = "Thinking Clearly, Talking Fast: Concept-Guided Non-Autoregressive Generation for Open-Domain Dialogue Systems",
        author = "Zou, Yicheng  and Liu, Zhihua  and Hu, Xingwu  and Zhang, Qi",
        booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
        month = nov,
        year = "2021",
        address = "Online and Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.emnlp-main.169",
        pages = "2215--2226"
    }
