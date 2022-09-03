# FGP-NLS
This repository contains files for symbolic regression with filtered genetic programming.

## Gettingb Started
### Prerequisites
The following prerequisites are needed:  
[Python 3.9](https://www.python.org/downloads/release/python-390/)  
[deap](https://github.com/DEAP/deap)  
[scipy](https://github.com/scipy/scipy)  
[sympy](https://github.com/sympy/sympy)  

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
![result pic](https://github.com/takakikatsushi/FGP-NLS/blob/main/Codes/result2/002_best_model_expr.png?raw=true)

## Authors  
  

## License  
[https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)
