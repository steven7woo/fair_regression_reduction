# fair_regression
Implementation for a reduction-based algorithm for fair regression.



If you find thie repository useful for your research, please consider citing our work:

```
@inproceedings{ADW19,
  author    = {Alekh Agarwal and
               Miroslav Dud{\'{\i}}k and
               Zhiwei Steven Wu},
  editor    = {Kamalika Chaudhuri and
               Ruslan Salakhutdinov},
  title     = {Fair Regression: Quantitative Definitions and Reduction-Based Algorithms},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning,
               {ICML} 2019, 9-15 June 2019, Long Beach, California, {USA}},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {120--129},
  publisher = {{PMLR}},
  year      = {2019},
  url       = {http://proceedings.mlr.press/v97/agarwal19d.html}
}
```





### Requirements
To run the code the following packages need to be installed:
- Gurobi solver. Avaiable at: http://www.gurobi.com/index
- Python package gurobipy. Avaiable with Anaconda: http://conda.anaconda.org/gurobi
- Python XGB package. This can be installed with "pip install xgboost"



### Usage
Main usage: run script run_exp.py to reproduce results in the paper.
