# Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation

This folder contains the code for the paper [《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》](https://arxiv.org/abs/1905.05621)



## Requirements

1. pip packages
    - pytorch>=0.4.0
    - torchtext>=0.4.0
    - nltk
    - fasttext==0.8.3
    - pypi-kenlm
    - tqdm
    
2. LM for evaluator
    - Download https://www.csie.ntu.edu.tw/~b05902064/ppl_yelp.binary
    - Put `ppl_yelp.binary` in the folder `evaluator`

## Usage

The hyperparameters for the Style Transformer can be found in `main.py` or with `python main.py -h`.

- To run task 1, use the command:
    ```shell
    python main.py -cyc_factor 0.0
    ```
- To run task 2, use the command:
    ```shell
    SAVE=./save/Feb15141010/ckpts/2000 # just example
    python main.py -F_pretrain_iter 0 \
    -cyc_factor 0.5 -temp 0.5 \
    -preload_F ${SAVE}_F.pth \
    -preload_D ${SAVE}_D.pth
    ```
You can modify other parameters to suit your need.

To evaluation the model, we used Fasttext,  NLTK and KenLM toolkit to evaluate the style control, content preservation and fluency respectively. The evaluation related files for the Yelp dataset are placed in the ''evaluator'' folder. 

See Requirements 2. for the file `ppl_yelp.binary`.


## Outputs

You can find the outputs of the original model in the "outputs" folder.
