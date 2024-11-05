import docplex
from eml.backend import cplex_backend
from eml.tree.reader.sklearn_reader import read_sklearn_tree
from eml.tree import embed
from docplex.mp.model_reader import ModelReader
from configdb import ConfigDB
from optimization_request import OptimizationSolution


def hada(db: ConfigDB,
         request,
         models,
         var_bounds,
         robust_coeff):
    """
    Implement HADA:
        1. Declare variables and basic constraints
        2. Embed predictive models
        3. Declare user-defined constraints and objective
        4. Solve the model and output an optimal matching (hw-platform, alg-configuration)

    PARAMETERS
    ---------
    xxx [yyy] : a transprecision computing algorithm {saxpy, convolution, correlation, fwt}
    xxx [yyy]: type {min, max} and target

    RETURN
    ------
    sol [dict]: optimal solution found
    mdl [docplex.mp.model.Model]: final optimization model
    """

    ####### MODEL #######
    bkd = cplex_backend.CplexBackend()
    mdl = docplex.mp.model.Model("HADA")
    # mdl.parameters.mip.tolerances.integrality = 0.0

    hws = db.get_hws(request.algorithm)
    targets = set(list(request.user_constraints.get_constraints().keys()) + [request.target])
    hyperparams = db.get_hyperparams(request.algorithm)

    # Retrieve variable types, assuming that price is always a float
    cplex_type = {'bin': mdl.binary_vartype, 'int': mdl.integer_vartype, 'float': mdl.continuous_vartype}
    var_type = db.get_type_per_var(request.algorithm)
    var_type['price'] = 'float'
    var_type = {var: cplex_type[var_type[var]] for var in var_type.keys()}

    ####### VARIABLES #######
    # A binary variable for each hw, specifying whether this hw is selected or not
    for hw in hws:
        mdl.binary_var(name=f"b_{hw}")

    ml_var = {}

    # A variable for each hyperparameter, whose type matches the hyperparameter's type
    # If the hyperparameter is non-continuous, it also requires an auxiliary continuous variable and 
    # an integrality constraint: the auxiliary variable is used as input to the predictive models (emllib 
    # accepts only continuous variables), the integrality (equality) constraint is used to convert the 
    # auxiliary variable back into the binary/integer one
    for hyperparam in hyperparams:
        mdl.var(name=hyperparam,
                vartype=var_type[hyperparam],
                lb=var_bounds[hyperparam]['lb'],
                ub=var_bounds[hyperparam]['ub'])
        ml_var[hyperparam] = hyperparam
        if var_type[hyperparam] != mdl.continuous_vartype:
            mdl.var(name=f"auxiliary_{hyperparam}",
                    vartype=mdl.continuous_vartype,
                    lb=var_bounds[hyperparam]['lb'],
                    ub=var_bounds[hyperparam]['ub'])
            mdl.add_constraint(mdl.get_var_by_name(hyperparam) == mdl.get_var_by_name(f"auxiliary_{hyperparam}"),
                               ctname=f"{hyperparam}_integrality_constraint")
            ml_var[hyperparam] = f'auxiliary_{hyperparam}'

    # A variable for each target and hw, whose type matches the target's type. 
    # Also in this case, if the target is non-continuous, it requires auxiliary variables and constraints
    for target in targets:
        for hw in hws:
            mdl.var(name=f"{hw}_{target}",
                    vartype=var_type[target],
                    lb=var_bounds[target]['lb'],
                    ub=var_bounds[target]['ub'])
            ml_var[f'{hw}_{target}'] = f'{hw}_{target}'
            if var_type[target] != mdl.continuous_vartype:
                mdl.var(name=f"auxiliary_{hw}_{target}",
                        vartype=mdl.continuous_vartype,
                        lb=var_bounds[target]['lb'],
                        ub=var_bounds[target]['ub'])
                mdl.add_constraint(
                    mdl.get_var_by_name(f'{hw}_{target}') == mdl.get_var_by_name(f"auxiliary_{hw}_{target}"),
                    ctname=f"{hw}_{target}_integrality_constraint")
                ml_var[f'{hw}_{target}'] = f'auxiliary_{hw}_{target}'

    ####### CONSTRAINTS ######
    # HW Selection Constraint, enabling the selection of a single hw platform
    mdl.add_constraint(mdl.sum(mdl.get_var_by_name(f"b_{hw}") for hw in hws) == 1, ctname="hw_selection")

    # Empirical Constraints: embed the predictive models into the system (through emllib)
    for target in targets:
        # target price is not predicted, but indicated by the hw provider: it does not require any
        # dedicated predictive model
        if target == "price":
            continue
        # time and memory depend on both the hw and the algorithm configuration: each of them requires three 
        # dedicated predictive models
        for hw in hws:
            model = models.get_model(request.algorithm, hw, target)
            model = read_sklearn_tree(model)
            for i, hyperparam in enumerate(hyperparams):
                model.update_lb(i, var_bounds[hyperparam]['lb'])
                model.update_ub(i, var_bounds[hyperparam]['ub'])
            embed.encode_backward_implications(
                bkd=bkd, mdl=mdl,
                tree=model,
                tree_in=[mdl.get_var_by_name(ml_var[hyperparam]) for hyperparam in hyperparams],
                tree_out=mdl.get_var_by_name(ml_var[f"{hw}_{target}"]),
                name=f"DT_{hw}_{target}")

    # Handling non-estimated target (price) and robustness coefficients: 
    # 1.Equality constraints, fixing each price variable hw_price to the usage price of the corresponding hw,
    # as required by the hw provider
    if 'price' in targets:
        for hw in hws:
            mdl.add_constraint(mdl.get_var_by_name(f"{hw}_price") == request.hws_prices.get_prices_per_hw()[hw],
                               ctname=f"{hw}_price")

    # 2. If no robustness is required, fix all coefficients to 0 
    if robust_coeff is None:
        robust_coeff = {(hw, target): 0
                        for hw in hws
                        for target in request.user_constraints.get_constraints()}

    # User-defined constraints, bounding the performance of the algorithm, as required by the user
    for target in request.user_constraints.get_constraints():
        for hw in hws:
            if request.user_constraints.get_constraints()[target][0] == "leq":
                mdl.add_indicator(mdl.get_var_by_name(f"b_{hw}"),
                                  mdl.get_var_by_name(f"{hw}_{target}") <=
                                  request.user_constraints.get_constraints()[target][1] - robust_coeff[(hw, target)], 1,
                                  name=f"user_constraint_{target}_{hw}")
            elif request.user_constraints.get_constraints()[target][0] == "geq":
                mdl.add_indicator(mdl.get_var_by_name(f"b_{hw}"),
                                  mdl.get_var_by_name(f"{hw}_{target}") >=
                                  request.user_constraints.get_constraints()[target][1] + robust_coeff[(hw, target)], 1,
                                  name=f"user_constraint_{target}_{hw}")
            elif request.user_constraints.get_constraints()[target][0] == "eq":
                mdl.add_indicator(mdl.get_var_by_name(f"b_{hw}"),
                                  mdl.get_var_by_name(f"{hw}_{target}") >=
                                  request.user_constraints.get_constraints()[target][1] - robust_coeff[(hw, target)], 1,
                                  name=f"user_constraint_{target}_{hw}_1")
                mdl.add_indicator(mdl.get_var_by_name(f"b_{hw}"),
                                  mdl.get_var_by_name(f"{hw}_{target}") <=
                                  request.user_constraints.get_constraints()[target][1] + robust_coeff[(hw, target)], 1,
                                  name=f"user_constraint_{target}_{hw}_2")

    ##### OBJECTIVE #####
    if request.opt_type == "min":
        mdl.minimize(
            mdl.sum(mdl.get_var_by_name(f"{hw}_{request.target}") * mdl.get_var_by_name(f"b_{hw}") for hw in hws))
    else:
        mdl.maximize(
            mdl.sum(mdl.get_var_by_name(f"{hw}_{request.target}") * mdl.get_var_by_name(f"b_{hw}") for hw in hws))

    ##### SOLVE #####
    sol = mdl.solve()

    solution = None
    if sol:
        for hw in hws:
            if round(sol[f'b_{hw}']) == 1:
                chosen_hw = hw
                break
        targets_values = {
            target: round(sol[f"{chosen_hw}_{target}"]) if var_type[target] != mdl.continuous_vartype else sol[
                f"{chosen_hw}_{target}"] for target in targets}
        hyperparams_values = {
            hyperparam: round(sol[hyperparam]) if var_type[hyperparam] != mdl.continuous_vartype else sol[hyperparam]
            for hyperparam in hyperparams}

        # solution = {'chosen_hw': chosen_hw, 'hyperparams': hyperparams_values, 'targets': targets_values}
        solution = OptimizationSolution(chosen_hw, hyperparams_values, targets_values)

    return solution
