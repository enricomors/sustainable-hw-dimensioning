import os
import json
import requests
from urllib.parse import urljoin


class ConfigDB:
    """Exposes information stored in the JSON configs (one config per algorithm/hardware pair)."""

    @classmethod
    def from_local(cls, path):
        """Initialize ConfigDB using local configs.

        Args:
            path (str): local path containing the configs.

        Returns:
            ConfigDB: instance of ConfigDB.
        """

        fnames = [os.path.join(path, fname) for fname in sorted(os.listdir(path))]
        algo_hw_couples = []
        configs = []

        # expected fnames: <algorithm>_<hw>.csv
        for fname in fnames:
            algorithm, part = fname.split('_')
            hw = part.split('.')[0]
            algo_hw_couples.append((algorithm, hw))
            configs.append(json.load(open(fname)))

        return cls(configs, algo_hw_couples)

    @classmethod
    def from_remote(cls, address):
        """Initialize ConfigDB using remote configs (VM storage ervice).

        Args:
            address (str): complete URL relative to the service that handles the configs.

        Returns:
            ConfigDB: instance of ConfigDB.
        """
        # test availability
        # requests.head(address)

        # getting list of config files
        configs_url = urljoin(address, '/configs')
        algo_hw_couples = [(config['algorithm'], config['hw'])
                           for config in requests.request('GET', configs_url).json()['configs']]

        configs = []
        # getting the actual configs
        for (algorithm, hw) in algo_hw_couples:
            algo_hw_url = urljoin(address, f'/configs/{algorithm}/{hw}')
            config = json.loads(requests.request('GET', algo_hw_url).content)
            configs.append(config)

        return cls(configs, algo_hw_couples)

    def __init__(self, configs, algo_hw_couples):
        """Initializes ConfigDB.

        Args:
            configs (list[dict]): list of configs, with each configs being represented as a dict.
            algo_hw_couples (list[tuple[str,str]]): list of (algorith_id, hardware_id) couples, corresponding, in order, to the configs.
        
        Raises:
            AttributeError: Hyperparameters and/or Targets not matching across different hardware given the same algorithm.
        """

        # scan path
        # self.fnames = [os.path.join(path, fname) for fname in sorted(os.listdir(path))]

        # dictionary with the name of algorithms as keys and values structured like this:
        # {
        #    'hyperparams': {'var_0': {'type': 'int', 'LB': None, 'UB': None},
        #                    'var_1': {'type': 'int', 'LB': None, 'UB': None}},
        #    'targets': {'time': {'LB': None, 'UB': None},
        #                'memory': {'LB': None, 'UB': None}},
        #    'hws': {'vm': None,
        #            'pc': None, 
        #            'g100': None}
        # }
        self.configs = configs
        self.algo_hw_couples = algo_hw_couples
        self.db = {}

        for config, (algorithm, hw) in zip(self.configs, self.algo_hw_couples):
            # load JSON files
            # config = json.load(open(fname))

            # checking types for all fields
            self.__check_json(algorithm, hw, config)

            # internal db structure
            hyperparams = {hyperparam['ID']: {'type': hyperparam['type'],
                                              'description': hyperparam['description'],
                                              'LB': hyperparam['LB'],
                                              'UB': hyperparam['UB']}
                           for hyperparam in config['hyperparams']}

            # 'type': target['type'],
            targets = {target['ID']: {'description': target['description'],
                                      'LB': target['LB'],
                                      'UB': target['UB']}
                       for target in config['targets']}

            # checking for overlap of names among hyperparams and targets
            if set.intersection(set(hyperparams), set(targets)):
                raise AttributeError(f'Names of hyperparams and targets must not overlap.')

            # checking consistency across hws for a given algorithm
            if config['name'] not in self.db:
                self.db[config['name']] = {'hyperparams': hyperparams,
                                           'targets': targets,
                                           'hws': {config['HW_ID']: config['HW_price']}}
            else:
                # checking consistency of hyperparameters across hws for a given algorithm
                if self.db[config['name']]['hyperparams'] != hyperparams:
                    raise AttributeError(
                        f'Hyperparameters not matching for algorithm {config["name"]} on different hws.')
                # checking consistency of targets across hws for a given algorithm
                if self.db[config['name']]['targets'] != targets:
                    raise AttributeError(f'Targets not matching for algorithm {config["name"]} on different hws.')

                # TODO (eventually): check consistency of HW prices (suggested in config) for a given HW across all algorithms.
                # Not needed; prices could be different for same hw and different algorithms (e.g. different contracts) 

                # just adding the new HW and its price, the rest must be the same across hws for the given algorithm.
                self.db[config['name']]['hws'][config['HW_ID']] = config['HW_price']

    def get_algorithms(self):
        """Get list of all available algorithms."""
        return list(self.db.keys())

    def get_hyperparams(self, algorithm):
        """Get list of hyperparameters for a given algorithm."""
        return list(self.db[algorithm]['hyperparams'].keys())

    def get_targets(self, algorithm):
        """Get list of targets for a given algorithm."""
        # price is the only "special" target, with possibly different handling
        return list(self.db[algorithm]['targets'].keys()) + ['price']

    def get_hws(self, algorithm):
        """Get list of hardware platforms for a given algorithm."""
        return list(self.db[algorithm]['hws'].keys())

    def get_prices(self, algorithm):
        """Get list of hardware prices for a given algorithm."""
        return list(self.db[algorithm]['hws'].values())

    def get_prices_per_hw(self, algorithm):
        """Get dict HW_name:price for all hws found for a given algorithm."""
        return self.db[algorithm]['hws']

    def get_lb_per_var(self, algorithm):
        """Get LBs for all variables (hyperparameters and targets)."""
        lb_per_var = {}

        for var in self.db[algorithm]['hyperparams']:
            lb_per_var[var] = self.db[algorithm]['hyperparams'][var]["LB"]

        for var in self.db[algorithm]['targets']:
            lb_per_var[var] = self.db[algorithm]['targets'][var]["LB"]

        return lb_per_var

    def get_ub_per_var(self, algorithm):
        """Get UBs for all variables (hyperparameters and targets)."""
        ub_per_var = {}

        for var in self.db[algorithm]['hyperparams']:
            ub_per_var[var] = self.db[algorithm]['hyperparams'][var]["UB"]

        for var in self.db[algorithm]['targets']:
            ub_per_var[var] = self.db[algorithm]['targets'][var]["UB"]

        return ub_per_var

    def get_description_per_var(self, algorithm):
        """Get description for all variables (hyperparameters and targets)."""
        description_per_var = {}

        for var in self.db[algorithm]['hyperparams']:
            description_per_var[var] = self.db[algorithm]['hyperparams'][var]["description"]

        for var in self.db[algorithm]['targets']:
            description_per_var[var] = self.db[algorithm]['targets'][var]["description"]

        return description_per_var

    def get_type_per_var(self, algorithm):
        """Get type for all variables (hyperparameters and targets)."""
        type_per_var = {}

        for var in self.db[algorithm]['hyperparams']:
            type_per_var[var] = self.db[algorithm]['hyperparams'][var]["type"]

        # assumption: targets are always continuous
        for var in self.db[algorithm]['targets']:
            type_per_var[var] = 'float'

        return type_per_var

    def __check_json(self, algorithm, hw, config):
        """Checks that the fields in the JSON configs are present and of of the expected types."""
        try:
            # checking algorithm
            if type(config['name']) is not str:
                AttributeError('Algorithm name must be a string')

            # checking hardware
            if type(config['HW_ID']) is not str:
                AttributeError('Hardware platform name must be a string')

            if config['HW_price'] is not None and type(config['HW_price']) not in [int, float]:
                raise AttributeError("Hardware platform price must be a number or None")

            # checking hyperparams
            for hyperparam in config['hyperparams']:
                if type(hyperparam['ID']) is not str:
                    raise AttributeError(f'ID of hyperparameters must be strings')

                if hyperparam['description'] is not None and type(hyperparam['description']) is not str:
                    raise AttributeError("Hyperparameter description must be a string")

                if hyperparam['type'] not in ['bin', 'int', 'float']:
                    raise AttributeError("Hyperparameter type must be 'bin', 'int' or 'float'")

                if hyperparam['UB'] is not None and type(hyperparam['UB']) not in [int, float]:
                    raise AttributeError("Hyperparameter upper bound must be a number or None")
                if hyperparam['LB'] is not None and type(hyperparam['LB']) not in [int, float]:
                    raise AttributeError("Hyperparameter lower bound must be a number or None")

            # checking targets
            for target in config['targets']:
                if type(target['ID']) is not str:
                    raise AttributeError(f'ID of targets must be strings; config: ({algorithm}, {hw}')

                if target['description'] is not None and type(target['description']) is not str:
                    raise AttributeError("Target description must be a string")

                # if target['type'] not in ['bin', 'int', 'float']:
                #    raise AttributeError("Targets type must be 'bin', 'int' or 'float'")

                if target['UB'] is not None and type(target['UB']) not in [int, float]:
                    raise AttributeError("Targets upper bound must be a number or None")
                if target['LB'] is not None and type(target['LB']) not in [int, float]:
                    raise AttributeError("Targets lower bound must be a number or None")

        except AttributeError as e:
            print(f'Error in config ({algorithm}, {hw})')
            raise e
