This repository contains code for the paper:

[Probabilistic Circuits for Variational Inference in Discrete Graphical Models](https://arxiv.org/abs/2010.11446)

```
"Probabilistic Circuits for Variational Inference in Discrete Graphical Models"
Andy Shih, Stefano Ermon
In Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS), 2020

@inproceedings{ShihEneurips20,
  author    = {Andy Shih and Stefano Ermon},
  title     = {Probabilistic Circuits for Variational Inference in Discrete Graphical Models},
  booktitle = {Advances in Neural Information Processing Systems 33 (NeurIPS)},
  month     = {december},
  year      = {2020},
}
```

Here are commands for running the experiments:

## Ising Models
```
python runising.py --loadgm=1000 --run=123 --mode=2 --n=4
python runising.py --loadgm=1000 --run=123 --mode=1 --n=8
python runising.py --loadgm=1000 --run=123 --mode=1 --n=16
python runising.py --loadgm=1000 --run=123 --mode=1 --n=32
```
, and repeat with loadgm=[1001,1002,1003].

## UAI Inference Competition
```
python runuai.py --run=123
```

## Contact
For questions, contact us at:

andyshih at cs dot stanford dot edu
