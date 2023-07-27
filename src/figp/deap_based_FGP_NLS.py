import decimal
import itertools
import operator
import os
import pickle
import random
import re
import time
import warnings
import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pydotplus
import sympy
from deap import algorithms, base, creator, gp, tools
import deap
# from IPython.display import Image
from scipy import optimize
from scipy.optimize import basinhopping, least_squares, leastsq, minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

from .deap_based_FGP_algorithms import FGP_NLS_algorithm
from .deap_based_func_node import Node_space, NumpyBasedFunction
# , add, cube, div, exp, ln, mul, sqrt, square, sub, protected_division, protected_sqrt, protected_ln, 
from .log_manager import table_log, txt_log
from .my_plot import line_plot, scatter_plot

warnings.simplefilter('ignore', RuntimeWarning)

CAN_USE_FUNC = ['add', 'sub', 'mul', 'div', 'sqrt', 'square', 'cube', 'ln', 'exp', 'protected_division', 'protected_ln', 'protected_sqrt']

def isfloat(_str):
    try:
        float(_str)
    except:
        return False
    return True


def get_depth_list(individual):
    depth = list()
    depth_pool = [0]
    for _node in individual:
        current_depth = depth_pool.pop()
        depth_pool.extend([current_depth+1]*_node.arity)
        depth.append(current_depth)
    return depth

def FV_filter(individual, function_group, function_filter, variable_filter):
    if function_filter or variable_filter:
        depth = get_depth_list(individual)
        func_pool, vals_pool = list(), list()
        com_flat = list(itertools.chain.from_iterable(function_group))
        old_d = -1
        for e, now_d in enumerate(depth):
            _node = individual[e]
            _name = individual[e].name
            _arity = individual[e].arity
            
            if old_d >= now_d:
                for _ in range(abs(old_d - now_d)+1):
                    func_pool.pop()
                    
            if function_filter:
                if _name in com_flat:
                    com_bool = [_name in c for c in function_group]
                    consider_pairs_n = [c for b, c in zip(com_bool, function_group) if b][0]
                    if sum([(n in func_pool) for n in consider_pairs_n])!=0:
                        return False, '=>>F-error'

            if variable_filter: 
                if _arity == 0: # check x or const node
                    if isinstance(_node.value, float): # True -> const node
                        pass
                    else:
                        if _name in vals_pool:
                            return False, '=>>V-error'
                        else:
                            vals_pool.append(_name)

            func_pool.append(_name)
            old_d = now_d

        state = ''
        if variable_filter:
            state += '=>>V-pass-' + '-'.join(vals_pool)
        else:
            state += '=>>V-none'
        if function_filter:
            state += '=>>F-pass'
        else:
            state += '=>>F-none'
        return True, state

    else:
        return True, '=>>F-none=>>V-none'


def D_filter(x_domain, y_domain, y_pred, equal, xydomain_filter):
    if xydomain_filter:
        if (x_domain is None or y_domain is None or y_pred is None):
            raise NameError(f'When xydomain_filter = True, x_domain needs a dataframe and y_domain needs a tuple of minimum and maximum values.\n x_domain = {type(x_domain)}, y_domain = {y_domain}')
        
        y_domain_min, y_domain_max = min(y_domain), max(y_domain)
        equal = ['=' if _e else '' for _e in equal]
        if eval(f'~np.all((y_domain_min <{equal[0]} y_pred)&(y_pred <{equal[0]} y_domain_max))'):
            return False, '=>>D-error'
        return True, '=>>D-pass'
    else:
        return True, '=>>D-none'

def C_only_filter(individual, constonly_filter):
    if constonly_filter:
        terminals=[isfloat(node.name) for node in individual if node.arity==0]
        if sum(terminals) == len(terminals):
            return False, '=>>C-error'
        else:
            return True, f'=>>C-pass({terminals})'
    else:
        return True, '=>>C-none'


def FVD_filter(
    individual, 
    function_filter = True, 
    variable_filter = True, 
    xydomain_filter = True, 
    constonly_filter = True,
    function_group=[ 
        ['sqrt', 'protected_sqrt'], 
        ['square', 'cube'], 
        ['ln', 'exp', 'protected_ln']],
    x_domain=None, 
    y_domain=None, 
    y_pred=None, 
    equal=(True, True)
    ):

    state = ''

    _bool, _state = C_only_filter(individual, constonly_filter)
    state += _state
    if _bool == False:
        return _bool, state
    
    _bool, _state = FV_filter(individual, function_group, function_filter, variable_filter)
    state += _state
    if _bool == False:
        return _bool, state

    _bool, _state = D_filter(x_domain, y_domain, y_pred, equal, xydomain_filter)
    state += _state
    if _bool == False:
        return _bool, state
    
    return True, state


class surveyed_individuals():
    def __init__(self, x_df):
        self.sympy_space_ = Node_space(x_df, func_list=CAN_USE_FUNC)
        self._surveyed_ind_pool_ = set([])
    
    def is_unobserved(self, ind, add_pool=True):
        expr = str(ind)
        if expr not in self._surveyed_ind_pool_:
            if add_pool: self._surveyed_ind_pool_.add(expr)
            return True
        else:
            return False

class Symbolic_Reg(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 population_size = 1000, 
                 generations     = 200, 
                 tournament_size = 5,
                 num_elite_select = 1,
                 max_depth       = 4,
                 function_set    = ['add', 'sub', 'mul', 'div', 'ln', 'exp', 'sqrt', 'square', 'cube'], 
                 metric          = 'mae',
                 p_crossover     = 0.7,
                 p_mutation      = 0.2,
                 random_state    = 0,
                 const_range     = (-1, 1),
                 init_max_trial  = 50000,
                 init_unique     = True,
                 var_max_trial   = 20,
                 function_filter = True, 
                 variable_filter = True, 
                 xydomain_filter = True,
                 constonly_filter= True,
                 x_domain        = None,
                 y_domain        = None,
                 domain_equal    = (True, True),
                 results_dir     = './deap_based_FGP_results'
                 ):
        """[summary]

        Args:
            population_size (int, optional): [description]. Defaults to 1000.
            generations (int, optional): [description]. Defaults to 200.
            tournament_size (int, optional): [description]. Defaults to 5.
            num_elite_select (int, optional): [description]. Defaults to 1.
            max_depth (int, optional): [description]. Defaults to 4.
            function_set (list, optional): [description]. Defaults to ['add', 'sub', 'mul', 'div', 'ln', 'log', 'sqrt'].
            metric (str, optional): [description]. Defaults to 'mae'.
            p_crossover (float, optional): [description]. Defaults to 0.7.
            p_mutation (float, optional): [description]. Defaults to 0.2.
            random_state (int, optional): [description]. Defaults to 0.
            const_range (tuple, optional): [description]. Defaults to (0, 1).
            x_domain ([type], optional): [description]. Defaults to None.
            y_domain ([type], optional): [description]. Defaults to None.
            results_dir (str, optional): [description]. Defaults to './results'.
        """
        
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.num_elite_select = num_elite_select
        self.max_depth = max_depth
        self.function_set = function_set
        self.metric = metric
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.random_state = random_state
        self.init_max_trial = init_max_trial
        self.init_unique = init_unique
        self.var_max_trial = var_max_trial
        self.const_range = const_range
        self.function_filter = function_filter
        self.variable_filter = variable_filter
        self.xydomain_filter = xydomain_filter
        self.constonly_filter= constonly_filter
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.domain_equal = domain_equal
        
        self.results_dir = results_dir

        self._n_time = 1
        self._c_node_ = 0

        self._x_columns = None


        random.seed(self.random_state)
        self.fit_x_ = None
        self.fit_y_ = None
        os.makedirs(self.results_dir, exist_ok=True)
        self.text_log_ = txt_log(file_name='000_GP_log_txt', save_path=self.results_dir)
        
        self._n_ind_generations = 0
        self._n_ind_gen_successes = 0
        self._warnings = 0
        
        self._can_use_func = ['add', 'sub', 'mul', 'div', 'sqrt', 'square', 'cube', 'ln', 'exp', 'protected_division', 'protected_ln', 'protected_sqrt']

    def fit(self, x, y):
        _start_gen = time.time()
        if isinstance(x, pd.DataFrame):
            self._x_columns = x.columns
            self.fit_x_ = x

        else:
            self._x_columns = [f'x{i}' for i in range(x.shape[1])]
            self.fit_x_ = pd.DataFrame(x, columns=self._x_columns)

        if self.x_domain is not None:
            if isinstance(self.x_domain, pd.DataFrame):
                if (self.x_domain.columns == self._x_columns).all():
                    pass
                else:
                    raise Exception('Column name mismatch (fit x <-> x_domain). fit X and X_domain must be of the same type.')
            else:
                if len(self._x_columns) == np.array(self.x_domain).shape[1]:
                    self.x_domain = pd.DataFrame(self.x_domain, columns=self._x_columns)
                else:
                    raise Exception('Column count mismatch. fit X and X_domain must be of the same type.')



        self.fit_y_ = y
        
        self._surveyed_individuals_ = surveyed_individuals(self.fit_x_)
        
        self.pset = gp.PrimitiveSet("MAIN", self.fit_x_ .shape[1])
        
        for i, x_name in enumerate(self.fit_x_.columns):
            p = {'ARG{}'.format(i):f'{x_name}'}
            self.pset.renameArguments(**p)

        if 'add' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.add, 2)
        if 'sub' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.sub, 2)
        if 'mul' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.mul, 2)
        if 'div' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.div, 2)
        if 'ln'  in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.ln, 1)
        if 'sqrt' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.sqrt, 1)
        if 'square' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.square, 1)
        if 'cube' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.cube, 1)
        if 'exp' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.exp, 1)
        if 'protected_division' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.protected_division, 2)
        if 'protected_sqrt' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.protected_sqrt, 1)
        if 'protected_ln' in self.function_set: self.pset.addPrimitive(NumpyBasedFunction.protected_ln, 1)

    
        # add initial constant to be optimized
        _run = True
        while _run:
            try:
                # self.pset.addEphemeralConstant(f'c_node_{self._c_node_}', 
                self.pset.addEphemeralConstant(f'{self._c_node_}', 
                                               lambda: random.uniform(self.const_range[0],
                                                                      self.const_range[1]))
                _run = False
            except:
                self._c_node_ += 1


        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        def _progress_bar():
            if int(self.population_size*0.01)*self._n_time == self._n_ind_gen_successes:
                if self._n_time%10 == 0:
                    print(f'{int(self._n_time)}%', end='')
                else:
                    print('|', end='')
                self._n_time += 1

        def filter_initIterate(container, generator, max_trial=self.init_max_trial, unique=self.init_unique, text_log=self.text_log_):
            for i in range(max_trial):
                ind = creator.Individual(generator())
                score = self._evalSymbReg(ind, self.fit_x_, self.fit_y_)
                self._n_ind_generations += 1
                
                if score[0] != np.inf:
                    if unique:
                        if self._surveyed_individuals_.is_unobserved(creator.Individual(ind)):
                            self._n_ind_gen_successes += 1
                            _progress_bar()
                            return container(ind)
                    else:
                        self._n_ind_gen_successes += 1
                        _progress_bar()
                        return container(ind)
                
            raise NameError(f'The maximum number of trials has been reached. \nNumber already generated : {self._n_ind_gen_successes}\nNumber of challenges : {self._n_ind_generations}')

        self.toolbox_ = base.Toolbox()
        self.toolbox_.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2) # gp.genHalfAndHalf https://deap.readthedocs.io/en/master/api/tools.html#deap.gp.genHalfAndHalf
        
        self.toolbox_.register("individual", filter_initIterate, creator.Individual, self.toolbox_.expr)
        # if (self.function_filter | self.variable_filter | self.xydomain_filter):
        #     self.toolbox_.register("individual", filter_initIterate, creator.Individual, self.toolbox_.expr)
        # else:
        #     # Normal generation without filter
        #     self.toolbox_.register("individual", tools.initIterate, creator.Individual, self.toolbox_.expr)
            
        self.toolbox_.register("population", tools.initRepeat, list, self.toolbox_.individual)
        self.toolbox_.register("compile", gp.compile, pset=self.pset)
        self.toolbox_.register("evaluate", self._evalSymbReg, x=self.fit_x_, y_true=self.fit_y_)
        self.toolbox_.register("select", tools.selTournament, tournsize=self.tournament_size)

        # gp.cxOnePoint : 1 point crossover
        # https://deap.readthedocs.io/en/master/api/tools.html?highlight=bloat#deap.gp.cxOnePoint
        self.toolbox_.register("mate", gp.cxOnePoint) 
        self.toolbox_.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))

        self.toolbox_.register("expr_mut", gp.genFull, min_=0, max_=2)

        # gp.mutUniform 
        # https://deap.readthedocs.io/en/master/api/tools.html?highlight=bloat#deap.gp.mutUniform
        self.toolbox_.register("mutate", gp.mutUniform, expr=self.toolbox_.expr_mut, pset=self.pset) 
        self.toolbox_.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth))
        
        self._time0 = time.time()

        print('Generation of initial generation')
        pop = self.toolbox_.population(n=self.population_size)
        print('\nGeneration complete')
        
        self.text_log_.print([f' {self._n_ind_gen_successes} [ ind ] / {self._n_ind_generations} [ trials ] (time : {(time.time() - self._time0)/60:.3f} min)\n'])
        
        self.hof = tools.HallOfFame(self.num_elite_select)
        
        pop, log = FGP_NLS_algorithm(population       = pop,
                            toolbox          = self.toolbox_, 
                            cxpb             = self.p_crossover,
                            mutpb            = self.p_mutation,
                            ngen             = self.generations,
                            halloffame       = self.hof,
                            num_elite_select = self.num_elite_select,
                            var_max_trial    = self.var_max_trial, 
                            check_func       = self._surveyed_individuals_,
                            text_log         = self.text_log_,
                            save_dir         = self.results_dir,
                            func_name        = self.function_set)
        
        self.expr = tools.selBest(pop, 1)[0]
        self.tree = gp.PrimitiveTree(self.expr)
        self.nodes, self.edges, self.labels = gp.graph(self.expr)
        self.log = log
        
        self.text_log_.print([f'FGP-NLS All Execution Time : {time.time() - _start_gen:.3f} s', 
                              f'FGP-NLS All Execution Time : {(time.time() - _start_gen)/60:.1f} min',
                              f'FGP-NLS All Execution Time : {(time.time() - _start_gen)/60/60:.1f} h',
                              f'Number of constant optimization warnings : {self._warnings}'
                              ])
        return self

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            if (self._x_columns == x.columns).all():
                pass
            else:
                raise Exception('Column name mismatch. fit X, predict X, and X_domain must be of the same type.')
        else:
            if len(self._x_columns) == np.array(x).shape[1]:
                x = pd.DataFrame(x, columns=self._x_columns)
            else:
                raise Exception('Column count mismatch. fit X, predict X, and X_domain must be of the same type.')
        y_pred = self._pred(x, self.expr)
        return y_pred
    
    def _pred(self, x, expr):
        func = self.toolbox_.compile(expr=expr)
        x_data = (x['{}'.format(i)] for i in list(x.columns))
        try:
            y_pred = func(*x_data)
        except:
            self._warnings += 1
            self.text_log_.print(['ERROR!! nan is included.', f'{str(expr)}', self.root])
            self.text_log_.print(self.temporary)
            self.text_log_.print(['ERROR!! _results'])
            self.text_log_.print(self.temporary2)
            
            y_pred = np.inf 
            
        if np.isscalar(y_pred):      # Avoid scalar errors.
            y_pred = pd.Series(np.full(x.shape[0], float(y_pred)))
        elif len(y_pred.shape) == 0: # Avoid errors due to singleton arrays.
            y_pred = y_pred.item()
            y_pred = pd.Series(np.full(x.shape[0], float(y_pred)))
        return y_pred
        
    def _evalSymbReg(self, individual, x, y_true):
        individual.state = ''

        # >>>>> func of const opt
        def _func(constants, x, y, individual, compiler, constant_nodes):
            _idx = 0
            for i in constant_nodes:
                if ~np.isnan(constants[_idx]):
                    cnode = copy.deepcopy(individual[i])
                    cnode.value = constants[_idx]
                    cnode.name  = str(constants[_idx])
                    
                    individual[i] = cnode
                _idx += 1

            _f = compiler(expr=individual)
            _x_data = (x['{}'.format(i)] for i in list(x.columns))
            _y = _f(*_x_data)
            if np.isscalar(_y):         # Avoid scalar errors.
                _y = np.full(x.shape[0], float(_y))
            elif len(_y.shape) == 0:    # Avoid errors due to singleton arrays.
                _y = _y.item()
                _y = np.full(x.shape[0], float(_y))
            elif type(_y) is pd.Series: # Set it to np.ndarray
                _y = _y.values
            y = y.values.reshape(len(_y))
            residual = y - _y
            
            return residual
        # <<<<< func of const opt
        
        filter_results1 = FVD_filter(individual, 
                                    function_filter = self.function_filter, 
                                    variable_filter = self.variable_filter, 
                                    xydomain_filter = False, 
                                    constonly_filter = self.constonly_filter,
                                    function_group=[ 
                                        ['sqrt'], 
                                        ['square', 'cube'], 
                                        ['ln', 'log', 'exp']],
                                    x_domain=None, 
                                    y_domain=None, 
                                    y_pred=None, 
                                    equal=self.domain_equal, 
                                    )
        if filter_results1[0]:
            pass
        else:
            individual.state = filter_results1[1]
            return np.inf,

        _is_const = [isfloat(n.name) for n in individual]
        # _is_const = [node.name == f'c_node_{self._c_node_}' for node in individual]
        
        if sum(_is_const):
            constant_nodes = [e for e, i in enumerate(_is_const) if i]
            constants0 = [individual[idx].value for idx in constant_nodes]
            self.temporary = constants0
            self.temporary2 = ['']
            if 0 < len(constants0):
                try:
                    _result=least_squares(_func, x0 = constants0, args=(x, y_true, individual, self.toolbox_.compile, constant_nodes), method='lm')
                    _idx = 0
                    if _result.status >= 2:
                        for i in constant_nodes:
                            cnode = copy.deepcopy(individual[i])
                            cnode.value = _result.x[_idx]
                            cnode.name  = str(_result.x[_idx])
                            # cnode.name  = f'c_node_{self._c_node_}'
                            individual[i] = cnode
                            _idx += 1
                        self.root = 'A'
                        self.temporary2 = [_result.x, _result.success, 'status', _result.status, _result.message]

                        if sum([_old==_new for _old, _new in zip(constants0, [individual[idx].value for idx in constant_nodes])]) == sum(_is_const):
                            opt_state = '=>>Copt-errorA'
                        else:
                            # opt_state = f'=>>Copt-pass({constants0}>>{_result.x})'
                            opt_state = f'=>>Copt-pass'
                        
                    else:
                        for i in constant_nodes:
                            cnode = copy.deepcopy(individual[i])
                            if ~np.isnan(constants0[_idx]):
                                cnode.value = constants0[_idx]
                                cnode.name  = str(constants0[_idx])
                            
                            individual[i] = cnode
                            _idx += 1
                        self.root = 'B'
                        opt_state = '=>>Copt-errorB'

                        
                except:
                    _idx = 0
                    for i in constant_nodes:
                        cnode = copy.deepcopy(individual[i])
                        if ~np.isnan(constants0[_idx]):
                            cnode.value = constants0[_idx]
                            cnode.name  = str(constants0[_idx])
                            
                        individual[i] = cnode
                        _idx += 1
                    self.root = 'C'
                    opt_state = '=>>Copt-errorC'
        else:
            _n = [node.name for node in individual]
            opt_state = f'=>>Copt-none({_n})'

        filter_results2 = FVD_filter(individual, 
                                    function_filter = False, 
                                    variable_filter = False, 
                                    xydomain_filter = self.xydomain_filter,
                                    constonly_filter = False,
                                    x_domain=self.x_domain, 
                                    y_domain=self.y_domain, 
                                    y_pred=self._pred(self.x_domain, individual), 
                                    equal=self.domain_equal, 
                                    )
        individual.state = filter_results1[1] + opt_state + filter_results2[1]
        if filter_results2[0]:
            pass
        else:
            return np.inf,
        
        y_pred = self._pred(self.fit_x_, individual)

        try:
            if self.metric == 'mae':
                error = mean_absolute_error(y_true, y_pred)
            elif self.metric == 'rmse':
                error = mean_squared_error(y_true, y_pred, squared=False)
            elif self.metric == 'mse':
                error = mean_squared_error(y_true, y_pred, squared=True)
            else:
                error = mean_absolute_error(y_true, y_pred)
        except:
            individual.state += '=>score-error'
            error = np.inf
        
        return error,
    
    def save_gen_metric_plot(self):
        log_file = pd.DataFrame(self.log)
        log_file.to_csv(f'{self.results_dir}/001_GP_log.tsv', sep='\t')
        
        line_plot(x_data=list(log_file.index), y_data=log_file.loc[:, ['score-min']], 
                  c_data=['b', 'r', 'g'], label_data=[f'{self.metric}-min'], 
                  xy_labels=['generation', f'{self.metric}'], figsize=(16,8), 
                  cmap='jet', color_bar=True, save_name=f'{self.results_dir}/001_GP_log_min', data_direction='column',
                  invert_xaxis=False, linewidth = 2.0, font_size=30, facecolor=None, vspan_data=None, line_style=['-', '--', ':'], )
        
        line_plot(x_data=list(log_file.index), y_data=log_file.loc[:, [f'score-min', f'score-med']], c_data=['b', 'r', 'g'], label_data=[f'{self.metric}-min', f'{self.metric}-med'], xy_labels=['generation', f'{self.metric}'], figsize=(16,8), 
                    cmap='jet', color_bar=True, save_name=f'{self.results_dir}/001_GP_log', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, facecolor=None, vspan_data=None, line_style=['-', '--', ':'])
        line_plot(x_data=list(log_file.index), y_data=log_file.loc[:, [f'score-std']], c_data=['g'], label_data=[f'{self.metric}-std'], xy_labels=['generation', f'{self.metric}'], figsize=(16,8), 
                    cmap='jet', color_bar=True, save_name=f'{self.results_dir}/001_GP_log_std', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, facecolor=None, vspan_data=None, line_style=['-', '--', ':'])
        line_plot(x_data=list(log_file.index), y_data=log_file.loc[:, ['unique_rate']], c_data=['m'], label_data=['unique_rate'], xy_labels=['generation', 'unique_rate'], figsize=(16,8), 
                    cmap='jet', color_bar=True, save_name=f'{self.results_dir}/001_GP_log_unique_rate', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, facecolor=None, vspan_data=None, line_style=['-', '--', ':'])
        return log_file

    def save_expr4word(self):
        sympy_space = Node_space(self.fit_x_, func_list=self._can_use_func)   
        expr = eval(str(self.expr), sympy_space.Nspace())
        expr4word = sympy.printing.octave.octave_code(expr).replace('./', '/').replace('.^', '^').replace('.*', '\\cdot ').replace('*', ' \\cdot ').replace('sqrt', '\\sqrt')
        f = open(f'{self.results_dir}/002_best_expr4word.txt', 'w', encoding='cp932')
        f.write(expr4word)
        f.close()
        
        non_e_expr4word = expr4word.replace('e+', '\\cdot 10^').replace('e-', '\\cdot 10^-')
        f = open(f'{self.results_dir}/002_best_expr4word_non_e.txt', 'w', encoding='cp932')
        f.write(non_e_expr4word)
        f.close()
        
        num_only = re.findall(r"\d+\.\d+", expr4word)
        for num in num_only:
            expr4word = expr4word.replace(num, Significant_figures(num, digit=3))
        
        f = open(f'{self.results_dir}/002_best_expr4word_3digits.txt', 'w', encoding='cp932')
        f.write(expr4word)
        f.close()
        
        non_e_expr4word = expr4word.replace('e+', '\\cdot 10^').replace('e-', '\\cdot 10^-')
        f = open(f'{self.results_dir}/002_best_expr4word_3digits_non_e.txt', 'w', encoding='cp932')
        f.write(non_e_expr4word)
        f.close()
        return expr4word
    
    def save_expr4tex(self, y_name=''):
        sympy_space = Node_space(self.fit_x_, func_list=self._can_use_func)
        # sympy.var(self.node_space_.symbol)
        expr = eval(str(self.expr), sympy_space.Nspace())
        expr_ = sympy.latex(expr)
        expr_la = expr_.replace('\\\\', '\\')
        f = open(f'{self.results_dir}/002_best_model.tex', 'w', encoding='cp932')
        f.write('\documentclass[a4paper, 12pt, fleqn]{tarticle}\n')
        f.write('\\begin{document}\n')
        f.write('\\begin{equation}\n')
        f.write(y_name + expr_la + '\n')
        f.write('\\end{equation}\n')
        f.write('\\end{document}\n')
        f.close()

        expr4png = sympy.latex(expr, mul_symbol='dot')#, min=-4, max=4)
        # expr4png = sympy.printing.latex(expr, mul_symbol='dot', min=-4, max=4)
        decimal_list = re.findall(r"\d\.\d+", expr4png)
        for string in decimal_list:
            if -1 < float(string) < 1:
                expr4png = expr4png.replace(string, f'{float(string):.4f}', 1)
            elif (-10 < float(string) <= -1 or 1 <= float(string) < 10):
                expr4png = expr4png.replace(string, f'{float(string):.3f}', 1)
            elif (-100 < float(string) <= -10 or 10 <= float(string) < 100):
                expr4png = expr4png.replace(string, f'{float(string):.2f}', 1)
            elif (-1000 < float(string) <= -100 or 100 <= float(string) < 1000):
                expr4png = expr4png.replace(string, f'{float(string):.1f}', 1)
            else:
                expr4png = expr4png.replace(string, f'{float(string):.0f}', 1)

        expr4png = y_name + expr4png
        tex = f'${expr4png}$'

        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.4, 0.4, rf'{tex}', size=30)
        plt.axis("off")
        plt.savefig(f'{self.results_dir}/002_best_model_expr.png', dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        return expr4png
    
    def save_node_analysis(self):
        log_file = pd.read_csv(f'{self.results_dir}/001_GP_node_analysis.tsv', sep='\t', index_col=0)
        _y_data = log_file.loc[:, list(self.fit_x_.columns)]
        line_plot(x_data=list(log_file.index), y_data=_y_data, c_data=list(range(len(_y_data.columns))), label_data=list(_y_data.columns), xy_labels=['generation', 'Content rate'], figsize=(16,8), 
                    color_bar=False, save_name=f'{self.results_dir}/001_GP_node_analysis_X', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, cmap='my_cmap', line_style=['-', '--', ':'])
        log_file = log_file.drop(list(self.fit_x_.columns), axis=1)
        line_plot(x_data=list(log_file.index), y_data=log_file, c_data=list(range(len(log_file.columns))), label_data=list(log_file.columns), xy_labels=['generation', 'Content rate'], figsize=(16,8), 
                    color_bar=False, save_name=f'{self.results_dir}/001_GP_node_analysis_func_all', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, cmap='my_cmap', line_style=['-', '--', ':'])
        log_file = log_file.drop('const/expr', axis=1)
        line_plot(x_data=list(log_file.index), y_data=log_file, c_data=list(range(len(log_file.columns))), label_data=list(log_file.columns), xy_labels=['generation', 'Content rate'], figsize=(16,8), 
                    color_bar=False, save_name=f'{self.results_dir}/001_GP_node_analysis_func', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, cmap='my_cmap', line_style=['-', '--', ':'])
        
        log_file = pd.read_csv(f'{self.results_dir}/001_GP_node_analysis_select.tsv', sep='\t', index_col=0)
        _y_data = log_file.loc[:, list(self.fit_x_.columns)]
        line_plot(x_data=list(log_file.index), y_data=_y_data, c_data=list(range(len(_y_data.columns))), label_data=list(_y_data.columns), xy_labels=['generation', 'Content rate'], figsize=(16,8), 
                    color_bar=False, save_name=f'{self.results_dir}/001_GP_node_analysis_X_select', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, cmap='my_cmap', line_style=['-', '--', ':'])
        log_file = log_file.drop(list(self.fit_x_.columns), axis=1)
        line_plot(x_data=list(log_file.index), y_data=log_file, c_data=list(range(len(log_file.columns))), label_data=list(log_file.columns), xy_labels=['generation', 'Content rate'], figsize=(16,8), 
                    color_bar=False, save_name=f'{self.results_dir}/001_GP_node_analysis_func_all_select', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, cmap='my_cmap', line_style=['-', '--', ':'])
        log_file = log_file.drop('const/expr', axis=1)
        line_plot(x_data=list(log_file.index), y_data=log_file, c_data=list(range(len(log_file.columns))), label_data=list(log_file.columns), xy_labels=['generation', 'Content rate'], figsize=(16,8), 
                    color_bar=False, save_name=f'{self.results_dir}/001_GP_node_analysis_func_select', data_direction='column', 
                    invert_xaxis=False, linewidth = 2.0, font_size=30, cmap='my_cmap', line_style=['-', '--', ':'])
    
    def save_gp_params(self):
        p_df = pd.Series(self.get_params())
        p_df.to_csv(f'{self.results_dir}/001_FilterGPSR_params.tsv', '\t')

    def save_expr(self):
        save_dict = dict(expr=str(self.expr), use_X=[node.value for node in self.expr if type(node)==deap.gp.Terminal])
        with open(f'{self.results_dir}/000_best_expr.json', 'w') as fp:
            json.dump(save_dict, fp)
    
    def save_all(self, y_name=''):
        self.save_gp_params()
        self.save_gen_metric_plot()
        self.save_expr4tex(y_name=y_name)
        self.save_expr4word()
        self.save_expr()
        self.save_node_analysis()
        # self.save_tree_pic()
        

def Significant_figures(num, digit=3):
    decimal.getcontext().prec = digit
    a = decimal.Decimal(num)
    b = decimal.Decimal(1)
    return str(a/b)

def output_score(   y_true_list,
                    y_pred_list,
                    data_name_list = ['train', 'test'],
                    save_name      = './'
                    ):
    metric = dict( R2 = r2_score, MAE = mean_absolute_error, RMSE = mean_squared_error)
    colname = []
    score = pd.DataFrame()
    for idx, _d in enumerate(y_true_list):
        data_name = data_name_list[idx]
        for key in metric:
            d_key = '{}_{}'.format(key, data_name)
            if key == 'RMSE':
                try:
                    score.at[0,d_key] = metric[key](y_true_list[idx], y_pred_list[idx], squared = False)
                except:
                    score.at[0,d_key] = np.inf
            elif key == 'R2':
                try:
                    score.at[0,d_key] = metric[key](y_true_list[idx], y_pred_list[idx])
                except:
                    score.at[0,d_key] = -np.inf
            else:
                try:
                    score.at[0,d_key] = metric[key](y_true_list[idx], y_pred_list[idx])
                except:
                    score.at[0,d_key] = np.inf
    score.to_csv('{}score.tsv'.format(save_name), sep='\t')
    


class ExprNameSpace():
    def __init__(self, X, func=['add', 'sub', 'mul', 'div', 'ln', 'sqrt', 'square', 'cube', 'exp']):
        self.X = X
        self.func = func
        self.nspace = {}
        self._make()
    
    def _make(self):
        self.nspace.update({_f:eval(f'NumpyBasedFunction.{_f}')for _f in self.func})
        self.nspace.update({xname:np.array(self.X[xname]).reshape(-1, 1) for xname in self.X.columns})


class LoadExpr():
    def __init__(self, expr):
        self.expr = expr

    def predict(self, X):
        ns = ExprNameSpace(X)
        return eval(str(self.expr), ns.nspace)

# old

# def D_filter(x_domain, y_domain, y_pred, equal, xydomain_filter):
#     if xydomain_filter:
#         if (x_domain is None or y_domain is None or y_pred is None):
#             raise NameError(f'When xydomain_filter = True, x_domain needs a dataframe and y_domain needs a tuple of minimum and maximum values.\n x_domain = {type(x_domain)}, y_domain = {y_domain}')

#         if sum(equal)==2:
#             if ~np.all((y_domain[0] <= y_pred) & (y_pred <= y_domain[1])):
#                 return False, '=>>D-error'
#         elif equal[0]:
#             if ~np.all((y_domain[0] <= y_pred) & (y_pred < y_domain[1])):
#                 return False, '=>>D-error'
#         elif equal[1]:
#             if ~np.all((y_domain[0] < y_pred) & (y_pred <= y_domain[1])):
#                 return False, '=>>D-error'
#         else:
#             if ~np.all((y_domain[0] < y_pred) & (y_pred < y_domain[1])):
#                 return False, '=>>D-error'
#         return True, '=>>D-pass'
        
#     else:
#         return True, '=>>D-none'


# def FVD_filter(
#     individual, 
#     function_filter = True, 
#     variable_filter = True, 
#     xydomain_filter = True, 
#     function_group=[ ['sqrt', 'protected_sqrt'], ['square', 'cube'], ['ln', 'exp', 'protected_ln']],
#     x_domain=None, 
#     y_domain=None, 
#     y_pred=None, 
#     equal=(True, True), 
#     constonly_filter = True):

#     depth = list()
#     depth_pool = [0]
#     for _node in individual:
#         current_depth = depth_pool.pop()
#         depth_pool.extend([current_depth+1]*_node.arity)
#         depth.append(current_depth)

#     F, V, filter
#     func_pool, vals_pool = list(), list()
#     com_flat = list(itertools.chain.from_iterable(function_group))
#     old_d = -1
    
#     for e, now_d in enumerate(depth):
#         _node = individual[e]
#         _name = individual[e].name
#         _arity = individual[e].arity
        
#         if old_d >= now_d:
#             for _ in range(abs(old_d - now_d)+1):
#                 func_pool.pop()
                
                
#         if function_filter:
#             if _name in com_flat:
#                 com_bool = [_name in c for c in function_group]
#                 consider_pairs_n = [c for b, c in zip(com_bool, function_group) if b][0]
#                 if sum([(n in func_pool) for n in consider_pairs_n])!=0:
#                     return False, 'F_filter'

#         if variable_filter: 
#             if _arity == 0: # check x or const node
#                 if isinstance(_node.value, float): # True -> const node
#                     pass
#                 else:
#                     if _name in vals_pool:
#                         return False, 'V_filter'
#                     else:
#                         vals_pool.append(_name)

#         func_pool.append(_name)
#         old_d = now_d
            
#     if xydomain_filter:
#         if (x_domain is None or y_domain is None or y_pred is None):
#             raise NameError(f'When xydomain_filter = True, x_domain needs a dataframe and y_domain needs a tuple of minimum and maximum values.\n x_domain = {type(x_domain)}, y_domain = {y_domain}')

#         if sum(equal)==2:
#             if ~np.all((y_domain[0] <= y_pred) & (y_pred <= y_domain[1])):
#                 return False, 'D_filter'
#         elif equal[0]:
#             if ~np.all((y_domain[0] <= y_pred) & (y_pred < y_domain[1])):
#                 return False, 'D_filter'
#         elif equal[1]:
#             if ~np.all((y_domain[0] < y_pred) & (y_pred <= y_domain[1])):
#                 return False, 'D_filter'
#         else:
#             if ~np.all((y_domain[0] < y_pred) & (y_pred < y_domain[1])):
#                 return False, 'D_filter'

#     if constonly_filter:
#         terminals=[isfloat(node.name) for node in individual if node.arity==0]
#         if sum(terminals) == len(terminals):
#             return False, 'const_only'
    
#     return True, 'all_pass'

# old 
# Symbolic_Reg() method
    # def save_tree_pic(self):
    #     x_name = self.fit_x_.columns
    #     nodes = list(range(len(self.expr)))
    #     edges = list()
    #     labels = dict()
    #     aritys = list()
    #     stack = []
    #     for idx, _node in enumerate(self.expr):
    #         aritys.append(_node.arity)
    #         if stack:
    #             edges.append((stack[-1][0], idx))
    #             stack[-1][1] -= 1
    #         if hasattr(_node, 'value'):
    #             if isfloat(_node.value):
    #                 labels[idx] = Significant_figures(num=_node.value, digit=3, use_E=True)
    #             else:
    #                 labels[idx] = x_name[int(_node.name.split('ARG')[1])]
    #         else:
    #             labels[idx] = _node.name
                
    #         stack.append([idx, _node.arity])
    #         while stack and stack[-1][1] == 0:
    #             stack.pop()
    #     tree_graph( nodes=nodes, 
    #                 edges=edges, 
    #                 labels=list(labels.values()),
    #                 aritys=aritys, 
    #                 dpi=400,
    #                 save_name=f'{self.results_dir}/002_best_model_tree', 
    #                 fill_color=['#00206b', '#aec7ff', '#ffc6d8'], # [node_color, descriptor_color, constant_color]
    #                 font_color=['#ffffff', '#000000', '#000000'])