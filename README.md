# BitGPT: A 1-bit version of the GPT Language model, inspired from [Andrej Karpathy's tutorial](https://github.com/karpathy/nanoGPT) on building a GPT from scratch.

BitGPT is an attempt at including the best practices of building a language model, while providing the user with as much accessibility and flexibility as possible. The 1-bit version is adapted from the paper 
[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764), originally created for the [LLama model](https://github.com/meta-llama/llama).

Like Andrej's tutorial, the model currently contains only a minimal decoder-only architecture (instead of an entire transformer), however, I hope to add some missing elements (like [RoPE](https://arxiv.org/abs/2104.09864)),
and other possible components and techniques to integrate the best performing parts of the current open-source models. The project is divided into organised categories, for both easy comparison and understanding. Each present (and hopefully, any future)
subdirectory will usually contain a `model.py`, defining the architecture, and a `train.py`, a script used for training the model. 

The training script will allow you to train everything from a model with as little as 50k parameters to ones over 1B parameters. 

## Training

Simply clone the repository

```
git clone https://github.com/bananya-ml/BitGPT
```

install the required dependencies (remember to use a virtual environment!)

```
$ pip install -r ./requirements.txt
```

Remember, this downloads a CUDA-enabled version of PyTorch, which can take a while. If you don't have a CUDA-capable system, or don't wish to use CUDA for whatever reason, simply install PyTorch using

```
$ pip install torch
```

Once the installation is complete, from the root directory, you can run

```
python ./gpt/train.py
```

or

```
python ./bitgpt/train.py
```

to train and save a model with the default settings. You might want to play around with the hyperparameters to balance speed and quality of your trained model.

| Name           | Description                                       |Type  | Default Values |
|----------------|---------------------------------------------------|------|----------------|
|--batch-size    |Batch size for training                            |int   |64              |
|--block-size    |Maximum context length for predictions             |int   |256             |
|--max-iters     |Number of epochs to train                          |int   |500000          |
|--eval-iters    |Number of batches used to estimate loss during eval|int   |200             |
|--eval-interval |Interval after which eval is performed             |int   |2000            |  
|--lr            |Learning rate                                      |float |6e-4            |
|--n-head        |Number of heads in the transformer architecture    |float |4               |
|--n-layer       |Number of layers of the transformer architecture   |float |4               |
|--n-embd        |Embedding dimension                                |float |384             |
|--dropout,--d   |Dropout value                                      |float |0.2             |
|--weight-decay  |Weight decay                                       |float |1e-1            |
|--decay-lr      |Flag for learning rate decay                       |bool  |True            |
|--warmup-iters  |Steps to warmup lr decay                           |int   |200             |
|--lr-decay-iters|Should be ~= max_iters per Chinchilla              |int   |500000          |
|--min-lr        |Should be learning rate/10 per Chinchilla          |int   |6e-5            |
|--wandb-log     |Logging using wandb (need to login to wandb first) |bool  |False           |
|--seed          |Random seed                                        |int   |1337            |
|--verbose       |set to 1 to see all recommended tunable parameters,|int   |0               |
|                |2 to see all parameters                            |      |                |


## Inference

Each directory containing a `model.py` will also contain a `generate.py` that can be used as 

```
python ./gpt/generate.py
```

from the root directory of the project. The following arguments can be used with the `generate.py` file to tune the output:

| Name           | Description                                                                |Type  | Default Values |
|----------------|----------------------------------------------------------------------------|------|----------------|
|--prompt        |Generation from the model follows the prompt                                |str   |''              |
|--num-samples   |Number of samples to generate                                               |int   |2               |
|--max-new-tokens|Maximum context length for predictions                                      |int   |2000            |
|--temperature   |1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions   |float |1.0             |
|--top-k         |Retain only the top_k most likely tokens, clamp others to have 0 probability|int   |200             |


+ **NOTE** According to an [FAQ released by Microsoft](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf), BitLinear layers require a low bit GEMM kernel during inference. No particular implementation of a kernel is provided by the paper, so we use an unofficial implementation of our own. Until such a time as the authors of [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) release an implementation, I will assume the kernel does not make a significant difference in the quality of inference.

## Data

The data directory contains 2 files: `shakespeare.txt` which contains 40,000 lines from William Shakespeare's writing and `astro.txt` which contains about 35,000 lines stripped from research papers around massive stars,
machine learning and spectroscopy. Either file can be chosen as the training data, and any other text file can be placed in the directory and be used as training material, after changing the relevant part of the code
to be use the custom dataset.

I will, in the future, try to add support for more types of datasets, e.g. an instruction dataset, as I add greater functionality to use the trained model, e.g. as a chatbot. 


## TODO

[ ] Rotatory Positonal Embedding [RoPE](https://arxiv.org/abs/2104.09864)\
[x] BPE encoding for training and inference\
[ ] Chat style inference

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## References

* <a id="Shuming2024"></a> Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei, The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits, [https://doi.org/10.1093/mnras/staa3540](https://doi.org/10.48550/arXiv.2402.17764)

* <a id="Jianlin2023"></a> Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu, RoFormer: Enhanced Transformer with Rotary Position Embedding, [https://doi.org/10.48550/arXiv.2104.09864](https://doi.org/10.48550/arXiv.2104.09864)
