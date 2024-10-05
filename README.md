# Large Language Model Empowered Embedding Generator for Sequential Recommendation

This is the implementation of the submission "Large Language Model Empowered Embedding Generator for Sequential Recommendation".

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: Tesla V100 32GB
  - Cuda: 10.2
  - Driver version: 440.95.01
  - CPU: Intel Xeon Gold 6133
- Software:
  - Python: 3.9.5
  - Pytorch: 1.12.0+cu102

You can pip install the `requirements.txt` to configure the environment.

## Preprocess the dataset

You can preprocess the dataset according to the following steps:

1. The raw dataset downloaded from website should be put into `/data/<yelp/fashion/beauty>/raw/`. The Yelp dataset can be obtained from [https://www.yelp.com/dataset](https://www.yelp.com/dataset). The fashion and beauty datasets can be obtained from [https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews).
2. Conduct the preprocessing code `data/data_process.py` to filter cold-start users and items. After the procedure, you will get the id file  `/data/<yelp/fashion/beauty>/hdanled/id_map.json` and the interaction file  `/data/<yelp/fashion/beauty>/handled/inter_seq.txt`.
3. Convert the interaction file to the format used in this repo by running `data/convert_inter.ipynb`.

## Stage 1: Supervised Contrastive Fine-Tuning (SCFT)

By SCFT, you can get a fine-tuned LLM and corresponding LLM embeddings.

1. Construct the item prompts by running the jupyter `/data/<yelp/fashion/beauty>/item_prompt.ipynb` and you can get the jsonline `/data/<yelp/fashion/beauty>/handled/item_str.jsonline` that saves the textual prompt of all items.
2. Download all the files and weights of LLaMA-7b from [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama) and put them to the folder `/resources/llama-7b/`
3. Run the SCFT bash, and the derived LLM embedding will be save to `/results/llm-emb/default.json`

```
bash experiments/<yelp/fashion/beauty>/scft/avg.bash
```

4. Then, run the script `/results/convert.ipynb` to convert and get the dimension-reduced LLM embedding. The derived embeddings will be saved to `/data/<yelp/fashion/beauty>/handled/default_pca.pkl`

⭐️ To ease the reproducibility of our paper, we also upload the derived LLM embeddings to this [link](https://ufile.io/2v2c6tqa). Please rename them before usage.

## Stage 2: Recommendation Adaptation Training (RAT)

By RAT, you can get the well-trained adapter and the final SRS models.

1. Pretrain a general SRS model to get the collaborative embeddings via:

```
bash experiments/<yelp/fashion/beauty>/rat/general.bash
```

2. You can reproduce all LLMEmb experiments by running the bash as follows:

```
bash experiments/<yelp/fashion/beauty>/rat/llmemb.bash
```

2. The log and results will be saved in the folder `log/`. The checkpoint will be saved in the folder `saved/`.

## Citation

If the code and the paper are useful for you, it is appreciable to cite our paper:

```
@article{liu2024large,
  title={Large Language Model Empowered Embedding Generator for Sequential Recommendation},
  author={Liu, Qidong and Wu, Xian and Wang, Wanyu and Wang, Yejing and Zhu, Yuanshao and Zhao, Xiangyu and Tian, Feng and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2409.19925},
  year={2024}
}
```
