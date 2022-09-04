# FGP-NLS
FGP-NLS is a symbolic regression using Filter-incorporated Genetic Programming with the constant optimization by Nonlinear Least Square. GP implementation is **DEAP** library and mathematical expressions are analyzed with the help of the **sympy** library.  
Three filters help generate interpretable expressions: Variable filter, Functional filter and Domain filter.

The main target of FGP-NLS is for QSAR/QSPR models. The three filters help find human-understandable mathematical expressions as a result of giving up the extremely precise expressions explaining a training data set. Details are found in `To be updated`. 

Reference: `To be updated`

## Gettingb Started
### Prerequisites
The following libraries are necessary on top of [Python 3.9](https://www.python.org/downloads/release/python-390/).

* [DEAP](https://github.com/DEAP/deap)  
* [scipy](https://github.com/scipy/scipy)  
* [sympy](https://github.com/sympy/sympy)  
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [pandas](https://github.com/pandas-dev/pandas)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)

  
### Installation
We recoomend you 

## How to use  
### Relationship learning
Usage is simple. Use it like this:
```
est = Symbolic_Reg( population_size=200,
                    generations=100,
                    x_domain=X,
                    y_domain=(0, 1),
                    results_dir='./result'
                    )
est.fit(X_train, y_train)
est.save_all()
```
population_size and generations are the population size and number of generations to repeat. Larger values are likely to improve accuracy, but take longer. For x_domain and y_domain, enter the expected X and y ranges. results_dir specifies the directory where the results are stored.

### Confirmation of results  
Please check "002_best_model_expr.png". The expression is saved.
![result pic](https://github.com/takakikatsushi/FGP-NLS/blob/main/Codes/result2/002_best_model_expr.png?raw=true)

## Authors  
Takaki Katushi: https://github.com/takakikatsushi

### Contributors to the FGP-NLS descriptors project:
Takaki Katsushi
Tomoyuki Miyao


## License  
The codes are licensed under Creative Commons Attribution 4.0 International License. See the LICENSE.md file for additional details.
[https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/)
