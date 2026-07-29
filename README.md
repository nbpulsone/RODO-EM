# RODO-EM: Robust Domain-Aware Entity Matching
Code repository for the paper _RODO-EM: Robust Domain-Aware Entity Matching_.

Entity Matching (EM) is a core task in data management that determines whether records from two or more data sources refer to the same real-world entity. Modern AI approaches, particularly those based on fine-tuning Pretrained Language Models (PLMs) and Large Language Models (LLMs), have achieved state-of-the-art performance on EM. Although these learned models achieve strong average performance, they often fall short on smaller or underrepresented domains (or categories). In this work, we present RoDo-EM, a robust optimization framework for domain-aware EM that combines robustness-aware loss functions, domain-aware sampling strategies, and regularization techniques to optimize a general-purpose EM model.

This work builds on the [DITTO](https://github.com/megagonlabs/ditto/tree/master) pipeline for fine-tuning a PLM for EM.
 
<img width="1536" height="1024" alt="rodoem_diagram" src="https://github.com/user-attachments/assets/eee8ce6a-1fa7-4694-b67a-e25192cc129a" />
  
## Requirements
1. Python 3.7.13
2. For full list of dependencies, see [requirements.txt](requirements.txt)

## Datasets
We derive the domain-partitioned datasets for our experiments from [The WDC Multi-Dimensional EM Benchmark](https://webdatacommons.org/largescaleproductcorpus/wdc-products/#toc5), [The WDC Product Corpus](https://webdatacommons.org/largescaleproductcorpus/v2/), [The Abt-Buy Dataset](https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution), and [The Walmart-Amazon Dataset](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md#walmart-amazon).
See the [data](data) folder for details, and [configs.json](configs.json) for the full list of available domain-partitioned datasets to run.

## Quick Start
To train and evaluate a model for Robust Domain-Aware EM type:

`./run_em_opt.sh <optimizer> <metric> <dataset> <output_directory_file_path>`

The available optimizers to run are: 
1. k (where k is an integer)
2. dro
3. parity
4. size

The available metrics to optimize for are:
1. worst
2. best
3. entropy
4. variance
5. ppvp (Predictive Positive Value Parity)
6. tprp (True Postivie Rate Parity)
7. smallest
8. biggest

Some examples: 

`./run_em_opt.sh 1 worst WDC/category_50un_50cc results` (worst-1 optimization on WDC Multi-Dimensional Dataset)

`./run_em_opt.sh dro worst WDC/products_medium results` (DRO optimization on WDC Products Corpus)

`./run_em_opt.sh parity variance AB/abtbuy_brand results` (Variance optimization optimization on Abt_Buy)

`./run_em_opt.sh parity ppvp WA/price results` (PPVP optimization on Walmart-Amazon)

The variables at the top of [run_em_opt.sh](run_em_opt.sh) script can be modified to change the hyperparameters (e.g. robust weight, ema smoothing, regularization, learning rate, batch size, etc.)

## Grid Search
To find the best set of hyperparmaters for a given optimizer-metric configuration, set the grids in [opt_grid.sh](opt_grid.sh) and run:

`./opt_grid.sh`

## Case Study
To run the case study from the paper, type:

`./run_case_study.sh`
