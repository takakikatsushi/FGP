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

## Authors  
  

## License  
[https://creativecommons.org/licenses/by/4.0/legalcode](https://creativecommons.org/licenses/by/4.0/legalcode)
