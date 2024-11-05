import os
import requests
from abc import ABC, abstractmethod
from collections import defaultdict
from io import StringIO
from urllib.parse import urljoin
import pandas as pd
from optimization_request import OptimizationRequest


def process_dataset(df):
    """
    Processes the dataframe to be compatible with HADA.

    TODO: app should be extended to take care of other attributes.
    """
    # Remove the first two columns, assuming they are "PV(kW)" and "Load(kW)"
    df = df.iloc[:, 2:]
    # Removes the gpu related columns
    df = df.drop(columns=['gpuAvg(MB)', 'gpuPeak(MB)', 'gpuEnergy(kW)'])
    # removes the geographic related columns (no numeric values)
    processed_df = df.drop(columns=['country', 'region'])
    # return processed dataframe
    return processed_df


class Datasets(ABC):
    """Class that handles all the operations on the datasets."""

    @abstractmethod
    def __init__(self):
        pass

    @classmethod
    def from_local(cls, db, data_path):
        """Initialize Datasets using local datasets.

        Args:
            db (ConfigDB): instance of ConfigDB.
            data_path (str): local path containing the datasets.

        Returns:
            Datasets: instance of Datasets.
        """
        return DatasetsLocal(db, data_path)

    @classmethod
    def from_remote(cls, db, address):
        """Initialize Datasets using remote datasets (VM storage ervice).

        Args:
            db (ConfigDB): instance of ConfigDB.
            address (str): complete URL relative to the service that handles the datasets.

        Returns:
            Datasets: instance of Datasets.
        """
        return DatasetsRemote(db, address)

    @abstractmethod
    def get_dataset(self, algorithm, hw) -> pd.DataFrame:
        """Returns the dataset (Pandas DataFrame) relative to the (algorithm, hw), if present."""
        pass

    def _check_dataset_consistency(self, df, algorithm, hw):
        """Checking the columns are the expected ones and that they are numericals."""

        df = process_dataset(df)

        hyperparams = self.db.get_hyperparams()
        data_targets = self.db.get_targets(algorithm)
        data_targets.remove('price')

        if set(df.columns) != set(hyperparams + data_targets):
            raise AttributeError(
                f'Columns in the dataset for algorithm {algorithm} and hardware {hw} are not the expected ones.')

        # from pandas.api.types import is_numeric_dtype
        type_per_var = self.db.get_type_per_var(algorithm)
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise AttributeError(
                    f'Column {column} in the dataset for algorithm {algorithm} and hardware {hw} is not numeric.')

            # checking consistency with vartype declared in configs: int, float or bin
            # float already checked: if it's numerical it can be interpreted as float
            expected_dtype = type_per_var[column]
            if expected_dtype == 'int' and not pd.api.types.is_integer_dtype(df[column]):
                raise ValueError(
                    f'Column {column} in the dataset for algorithm {algorithm} and hardware {hw} is expected to be integer, but has non-integer values.')
            elif expected_dtype == 'bin' and set(df[column].unique()) != {0, 1}:
                raise ValueError(
                    f'Column {column} in the dataset for algorithm {algorithm} and hardware {hw} is expected to be binary, but has non-binary values.')

    def extract_var_bounds(self, algorithm):
        """
        Compute upper and lower bounds of each variable.
        If UB/LB specified in configs, use that instead of extracting from data.

        Args:
            algorithm (str): algorithm for which we want to extract variable bounds.

        Returns:
            lb_per_var (dict): lower bound for each variable (hyperparameters and targets).
            ub_per_var (dict): upper bound for each variable (hyperparameters and targets).
        """
        # check if both UB and LB are specified in the configs
        # otherwise add to "missing_bounds"; if any extract from data and calculate those

        # retrieving LBs/UBs from configs
        lb_per_var = self.db.get_lb_per_var(algorithm)
        ub_per_var = self.db.get_ub_per_var(algorithm)

        # handling non-specified bounds by extracting them from data
        lb_missing_vars = [var for var, lb in lb_per_var.items() if lb is None]
        ub_missing_vars = [var for var, ub in ub_per_var.items() if ub is None]
        missing_vars = set(lb_missing_vars + ub_missing_vars)

        # at least one bound to be extracted
        if missing_vars:
            # read one HW config at a time
            # extract needed mins and max
            # take overall min of minima and max of maxima
            all_mins_per_var = defaultdict(list)
            all_maxes_per_var = defaultdict(list)

            for hw in self.db.get_hws(algorithm):

                dataset = self.get_dataset(algorithm, hw)

                for var in lb_missing_vars:
                    all_mins_per_var[var].append(dataset[var].min())
                for var in ub_missing_vars:
                    all_maxes_per_var[var].append(dataset[var].max())

            for var in lb_missing_vars:
                lb_per_var[var] = min(all_mins_per_var[var]).item()
            for var in ub_missing_vars:
                ub_per_var[var] = max(all_maxes_per_var[var]).item()

            # checking that dtypes of variables are compatible with the bounds
            type_per_var = self.db.get_type_per_var(algorithm)
            for var, dtype in type_per_var.items():
                var_lb = lb_per_var[var]
                var_ub = ub_per_var[var]
                if dtype == 'int':
                    if type(var_lb) is not int or type(var_ub) is not int:
                        raise ValueError(f'Bound for variable {var} is not of the expected type (int).')
                elif dtype == 'bin':
                    if (var_lb not in [0, 1]) or (var_ub not in [0, 1]):
                        raise ValueError(
                            f'Bound for variable {var} is not of the expected type (bin): it must be 0 or 1.')

        return lb_per_var, ub_per_var

    def get_var_bounds_all(self, request: OptimizationRequest):
        """
        Compute upper and lower bounds of each variable, including price.
        If UB/LB specified in configs, use that instead of extracting from data.

        Args:
            request (OptimizationRequest): instance of OptimizationRequest.

        Returns:
            var_bounds (dict): lower bound and upper bound for each variable, including price.
        """

        lb_per_var, ub_per_var = self.extract_var_bounds(request.algorithm)

        if request.target == 'price' or 'price' in request.user_constraints.get_constraints():
            lb_per_var['price'] = min(request.hws_prices.get_prices_per_hw().values())
            ub_per_var['price'] = max(request.hws_prices.get_prices_per_hw().values())

        var_bounds = {var: {'lb': lb_per_var[var], 'ub': ub_per_var[var]}
                      for var in lb_per_var}
        return var_bounds

    def get_robust_coeff(self, models, request):
        """
        Compute robustness coefficients for each predictive model, according to the specified robustness factor.

        Args:
            models (MLModels): object that handles ML models.
            request (OptimizationRequest): represents the user's request.

        Returns:
            robust_coeff (dict): robustness coefficient for each predictive model.
        """

        if request.robustness_fact or request.robustness_fact == 0:
            robust_coeff = {}
            for target in self.db.get_targets(request.algorithm):
                for hw in self.db.get_hws(request.algorithm):
                    # The target price is not estimated: it does not require any robustness coefficient 
                    if target == 'price':
                        robust_coeff[(hw, "price")] = 0
                    else:
                        dataset = self.get_dataset(request.algorithm, hw)
                        model = models.get_model(request.algorithm, hw, target)

                        dataset[f'{target}_pred'] = model.predict(
                            dataset[[col for col in dataset.columns if 'var' in col]])
                        dataset[f'{target}_error'] = (dataset[f'{target}'] - dataset[f'{target}_pred']).abs()
                        robust_coeff[(hw, target)] = dataset[f'{target}_error'].std() * dataset[
                            f'{target}_error'].quantile(request.robustness_fact)
            return robust_coeff
        else:
            return None


class DatasetsLocal(Datasets):
    """Handles datasets stored locally."""

    def __init__(self, db, data_path):
        self.db = db
        self.data_path = data_path

    def get_dataset(self, algorithm, hw):
        dataset_path = os.path.join(self.data_path, f'{algorithm}_{hw}.csv')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f'Dataset for ({algorithm}, {hw}) not found.')

        dataset = pd.read_csv(dataset_path)

        # checking if data complies to configs
        self._check_dataset_consistency(dataset, algorithm, hw)

        return dataset


class DatasetsRemote(Datasets):
    """Handles retrieval of datasets from the storage web service."""

    def __init__(self, db, address):
        self.db = db
        self.address = address

    def get_dataset(self, algorithm, hw):
        algo_hw_url = urljoin(self.address, f'/datasets/{algorithm}/{hw}')
        req = requests.request('GET', algo_hw_url)
        if req.status_code != 200:
            raise FileNotFoundError(f'Dataset for ({algorithm}, {hw}) not found.')
        csv_file = req.content

        dataset = pd.read_csv(StringIO(csv_file.decode('utf-8')))

        # checking if data complies to configs
        self._check_dataset_consistency(dataset, algorithm, hw)

        return dataset
