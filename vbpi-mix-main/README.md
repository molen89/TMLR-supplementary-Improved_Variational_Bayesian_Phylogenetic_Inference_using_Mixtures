# vbpi-mix
Pytorch Implementation of Variational Bayesian Phylogenetic Inference using Mixtures, this repository is based on [vbpi-torch](https://github.com/zcrabbit/vbpi-torch) but only focuses on the unrooted section


## Dependencies

* [Biopython](http://biopython.org)
* [bitarray](https://pypi.org/project/bitarray/)
* [dendropy](https://dendropy.org)
* [ete3](http://etetoolkit.org)
* [PyTorch](https://pytorch.org/)
* [rSPR](https://github.com/cwhidden/rspr)
* [wandb](https://wandb.ai)
* [scipy](https://scipy.org)
* [networkx](https://networkx.org/)

You can build and enter a conda environment with all of the dependencies built in using the supplied `environment.yml` file via:

```
conda env create -f environment.yml
conda activate vbpi-torch
```



## Running

Examples:

In the unrooted/ folder, 

First, sign in to the wandb account 


Running a mixture using two componenets with both testing and traning
```bash
python main.py --dataset DS1 \
  --train \
  --test \
  --wandb_group run \
  --wandb_mode online \
  --S 2
```

Running a mixture using two componenets and normalizing flows with both testing and traning
```bash
python main.py --dataset DS1 \
  --train \
  --test \
  --wandb_group run \
  --wandb_mode online \
  --S 2 \
  --use_nf \
  --flow_type realnvp
```
