# BitGPT: A 1-bit version of the GPT Language model, inspired from [Andrej Karpthay's tutorial](https://github.com/karpathy/nanoGPT) on building a GPT from scratch.

BitGPT is an attempt at including the best practices of building a language model, while providing the user with as much accessibility and flexibility as possible. The 1-bit version is adapted from the paper 
[The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764), originally created for the [LLama model](https://github.com/meta-llama/llama).

Like Andrej's tutorial, the model currently contains only a minimal decoder-only architecture (instead of an entire transformer), however, I hope to add some missing elements (like [RoPE](https://arxiv.org/abs/2104.09864)), OpenAI's BPE encoding
and other possible components to integrate the best components of the current open-source models. The project is divided into organised categories, for both easy comparison and understanding. Each present (and hopefully, any future)
subdirectory will usually contain a `model.py`, defining the architecture, and a `train.py`, a script used for training the model. 

The training script will allow you to train everything from a model with as little as 50k parameters to ones over 1B parameters. 

## Install

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

Once the installation is complete, run

```
python train.py
```

to train and save a model with the default settings. You might want to play around with the hyperparameters to balance speed and quality of your trained model.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Link to MIT License](https://opensource.org/licenses/MIT)

## References

* <a id="Shuming2024"></a> Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei, The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits, [https://doi.org/10.1093/mnras/staa3540](https://doi.org/10.48550/arXiv.2402.17764)

* <a id="Jianlin2023"></a> Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu, RoFormer: Enhanced Transformer with Rotary Position Embedding, [https://doi.org/10.48550/arXiv.2104.09864](https://doi.org/10.48550/arXiv.2104.09864)
