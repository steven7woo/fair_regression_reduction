# Fair Regression: Reduction-Based Algorithms

Implementation for a reduction-based algorithm for fair regression
subject to the constraint of demographic parity (also called statistical parity).


If you find thie repository useful for your research, please consider
citing our work:

```
@inproceedings{ADW19,
  author    = {Alekh Agarwal and
               Miroslav Dud{\'{\i}}k and
               Zhiwei Steven Wu},
  title     = {Fair Regression: Quantitative Definitions and Reduction-Based Algorithms},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning,
               {ICML} 2019, 9-15 June 2019, Long Beach, California, {USA}},
  year      = {2019},
  url       = {http://proceedings.mlr.press/v97/agarwal19d.html}
}
```
[arXiv link to this paper](https://arxiv.org/abs/1905.12843)


### Requirements
To run the code the following packages need to be installed:
- [Gurobi solver](http://www.gurobi.com/index).
- Python package gurobipy. Avaiable with [Anaconda](http://conda.anaconda.org/gurobi).
- Python XGB package. [Installation guide](https://xgboost.readthedocs.io/en/latest/build.html).


### Dataset
We include three datasets.
- Adult Income 
- LSAC National Longitudinal (Law School) 
- Communities and Crime 



### Usage
- To train a fair regression model, run exp_grad.py.
- Run run_exp.py to reproduce results in the paper.


### Bounded group loss
This implementation focuses on demographic parity. For fair regression
  with bounded group loss constraint, please see the implementation in
  [fairlearn
  library](https://fairlearn.github.io/user_guide/mitigation.html?highlight=bounded%20group).
