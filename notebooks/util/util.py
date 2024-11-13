#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import pickle
from abc import abstractmethod, ABC
from collections import defaultdict
from logging import ERROR
from urllib.parse import urljoin
from multiprocessing import Process, Manager

import docplex
import requests
from eml.backend import cplex_backend
from eml.tree.reader.sklearn_reader import read_sklearn_tree
from eml.tree import embed
from gurobipy import Model
from gurobipy import GRB
import numpy as np
from memory_profiler import profile
from memory_profiler import memory_usage
from sklearn.tree import DecisionTreeRegressor

import pandas as pd
import psutil
import os
import time
import resource
import gc
import logging
import csv


def parse_data_gme(fname):
    """parse GME market prices."""
    # prepare a list for the results
    prices_gme = []

    # open the input file
    with open(fname, 'r') as fin:
        reader = csv.reader(fin)
        for i, row in enumerate(reader):
            if i > 0:
                # each data point is a pair:
                # - the first item is an hour
                # - the second item is the grid electricity price
                if int(row[0]) > 23:
                    print('ERROR: invalid hour in: ' + str(row))
                else:

                    prices_gme.append(float(row[1]) / 1000)
    price_gme_96 = []
    gme = np.asarray(prices_gme)

    for i in range(24):
        price_gme_96.append(gme[i])
        price_gme_96.append(gme[i])
        price_gme_96.append(gme[i])
        price_gme_96.append(gme[i])

    # return the read data
    return price_gme_96


def online_ant(scenarios, instance, file):
    """
    Runs the Online Anticipate algorithm on a given instance.

    :param scenarios: Number of scenarios.
    :param instance: Instance id.
    :param file: path of .csv file containing the instances.
    """

    # timestamps n instances m scenarios ns
    n = 96
    par = 100
    parNorm = scenarios
    m = 1

    # price data from GME
    prices_path = os.path.join('../data/', 'PricesGME.csv')
    cGrid = parse_data_gme(prices_path)
    cGridSt = np.mean(cGrid)

    # capacities and prices
    listc = []
    objX = np.zeros((m, n))
    objTot = [None] * m
    avg_mem_usage = [] * m
    gpu_usage = [] * m

    listc, objList, runList, objFinal, runFinal, memFinal, cpuPercUsage, mem, cpuMax, memMax, memPerc, memPercMax = \
        [[] for _ in range(12)]

    run = 0
    ob = 0

    (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, cap, change, phi, notphi, pDiesel, pStorageIn, pStorageOut, pGridIn,
     pGridOut, tilde_cons) = [[None] * n for _ in range(20)]

    capMax = 1000
    inCap = 200
    cDiesel = 0.054
    eff = 0.9
    cGridS = 0.035
    cRU = 0.045
    pDieselMax = 1200
    runtime = 0
    phiX = 0
    ubStIn = 400
    ubStOut = 400
    ubGrid = 600
    trace_solutions = np.zeros((m, n, 9))

    # Read instances file
    file_path = os.path.join('../data/', file)
    instances = pd.read_csv(file_path)

    # instances pv
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

    pRenPV = [instances['PV(kW)'][instance] for i in range(m)]
    np.asarray(pRenPV)

    # instances load
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))
    totCons = [instances['Load(kW)'][instance] for i in range(m)]
    np.asarray(totCons)

    shift = np.load(os.path.join('../data/', 'realizationsShift.npy'))
    capX1 = 200

    for j in range(m):
        gc.collect(2)
        gc.set_debug(gc.DEBUG_SAVEALL)

        for i in range(n):
            # n2 = remaining step for scenarios
            n2 = n - i

            # ns = scenarios
            cap2 = np.zeros((scenarios, n2))
            pDiesel2 = np.zeros((scenarios, n2))
            pStorageIn2 = np.zeros((scenarios, n2))
            pStorageOut2 = np.zeros((scenarios, n2))
            pGridIn2 = np.zeros((scenarios, n2))
            pGridOut2 = np.zeros((scenarios, n2))
            runtime = 0

            tilde_cons_scen = np.zeros((scenarios, n2))
            pRenPV_scen = np.zeros((scenarios, n2))

            # Set up logging to a file
            logging.basicConfig(filename='gurobi_debug.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

            logging.info("Starting Gurobi model creation...")
            try:
                mod = Model("full_model")
                logging.info("Gurobi model created successfully.")
            except Exception as e:
                logging.error(f"Failed to create Gurobi model: {e}")

            try:
                # build variables and define bounds
                pDiesel[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pDiesel_" + str(i))
                pStorageIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageIn_" + str(i))
                pStorageOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pStorageOut_" + str(i))
                pGridIn[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGrid_" + str(i))
                pGridOut[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="pGrid_" + str(i))
                cap[i] = mod.addVar(vtype=GRB.CONTINUOUS, name="cap_" + str(i))
                change[i] = mod.addVar(vtype=GRB.INTEGER, name="change")
                phi[i] = mod.addVar(vtype=GRB.BINARY, name="phi")
                notphi[i] = mod.addVar(vtype=GRB.BINARY, name="notphi")
                logging.info("Variables created successfully")
            except Exception as e:
                logging.error(f"Failed to create variables: {e}")

            #################################################
            # Shift from Demand Side Energy Management System
            #################################################

            tilde_cons[i] = (shift[i] + totCons[j][i])

            ####################
            # Model constraints
            ####################

            try:
                mod.addConstr(notphi[i] == 1 - phi[i])
                mod.addGenConstrIndicator(phi[i], True, pStorageOut[i], GRB.LESS_EQUAL, 0)
                mod.addGenConstrIndicator(notphi[i], True, pStorageIn[i], GRB.LESS_EQUAL, 0)

                # power balance constraint
                mod.addConstr((pRenPV[j][i] + pStorageOut[i] + pGridOut[i] + pDiesel[i] - pStorageIn[i] - pGridIn[i] ==
                               tilde_cons[i]), "Power balance")

                # Storage cap
                if i == 0:
                    mod.addConstr(cap[i] == inCap)
                    mod.addConstr(pStorageIn[i] <= capMax - inCap)
                    mod.addConstr(pStorageOut[i] <= inCap)
                else:
                    mod.addConstr(cap[i] == capX + eff * pStorageIn[i] - eff * pStorageOut[i])
                    mod.addConstr(pStorageIn[i] <= capMax - capX)
                    mod.addConstr(pStorageOut[i] <= capX)

                mod.addConstr(pStorageIn[i] <= ubStIn)
                mod.addConstr(pStorageOut[i] <= ubStOut)

                mod.addConstr(cap[i] <= capMax)
                mod.addConstr(cap[i] >= 0)

                # Diesel and Net cap

                mod.addConstr(pDiesel[i] <= pDieselMax)
                mod.addConstr(pGridIn[i] <= ubGrid)

                # Storage mode change

                mod.addConstr(change[i] >= 0)

                mod.addConstr(change[i] >= (phi[i] - phiX))
                mod.addConstr(change[i] >= (phiX - phi[i]))

                logging.info("First set of constraints added successfully")
            except Exception as e:
                logging.error(f"Failed to add first set of constraints: {e}")

            try:

                # build objective (only Power with direct costs)
                c = (cGrid[i] * pGridOut[i] + cDiesel * pDiesel[i] + cGridSt * pStorageIn[i] - cGridS * pGridIn[i] + change[
                    i] * cRU)

                logging.info("Objective built successfully")
            except Exception as e:
                logging.error(f"Failed to build objective: {e}")

            try:
                # Second Stage variables
                pDiesel2 = mod.addVars(scenarios, n2, vtype=GRB.CONTINUOUS, name="pDiesel_")
                pStorageIn2 = mod.addVars(scenarios, n2, vtype=GRB.CONTINUOUS, name="pStorageIn_")
                pStorageOut2 = mod.addVars(scenarios, n2, vtype=GRB.CONTINUOUS, name="pStorageOut_")
                pGridIn2 = mod.addVars(scenarios, n2, vtype=GRB.CONTINUOUS, name="pGrid_")
                pGridOut2 = mod.addVars(scenarios, n2, vtype=GRB.CONTINUOUS, name="pGrid_")
                cap2 = mod.addVars(scenarios, n2, vtype=GRB.CONTINUOUS, name="cap_")
                change2 = mod.addVars(scenarios, n2, vtype=GRB.INTEGER, name="change")
                phi2 = mod.addVars(scenarios, n2, vtype=GRB.BINARY, name="phi")
                notphi2 = mod.addVars(scenarios, n2, vtype=GRB.BINARY, name="notphi")

                logging.info("Second stage variables added successfully")
            except Exception as e:
                logging.error(f"Failed to add Second stage variables: {e}")

            try:

                # Define Scenarios
                all_load_scen = np.load(os.path.join('../data/', 'outfileLoad.npy'))
                all_PV_scen = np.load(os.path.join('../data/', 'outfilePV.npy'))

                load_scen = np.zeros((scenarios, n))
                PV_scen = np.zeros((scenarios, n))

                np.random.seed(3)
                for s in range(scenarios):
                    load_scen[s] = np.random.choice(all_load_scen[s], n)

                for s in range(scenarios):
                    PV_scen[s] = np.random.choice(all_PV_scen[s], n)

                for s in range(scenarios):
                    for z in range(n2):
                        tilde_cons_scen[s, z] = load_scen[s][z]
                        pRenPV_scen[s, z] = PV_scen[s][z]

                logging.info("Scenarios defined successfully")
            except Exception as e:
                logging.error(f"Failed to define scenarios: {e}")

            try:
                # power balance constraint
                mod.addConstrs((pRenPV_scen[s, z] + pStorageOut2[s, z] + pGridOut2[s, z] + pDiesel2[s, z] -
                                pStorageIn2[s, z] - pGridIn2[s, z] == tilde_cons_scen[s, z]
                                for s in range(scenarios) for z in range(n2)), "Power balance")

                for s in range(scenarios):
                    for z in range(n2):
                        mod.addConstr(notphi2[s, z] == 1 - phi2[s, z])
                        mod.addGenConstrIndicator(phi2[s, z], True, pStorageOut2[s, z], GRB.LESS_EQUAL, 0)
                        mod.addGenConstrIndicator(notphi2[s, z], True, pStorageIn2[s, z], GRB.LESS_EQUAL, 0)

                        mod.addConstr(cap2[s, z] == capX1 + 0.9 * pStorageIn2[s, z] - 0.9 * pStorageOut2[s, z])

                        mod.addConstr(pStorageIn2[s, z] <= capMax - capX1)
                        mod.addConstr(pStorageIn2[s, z] <= capX1)

                        mod.addConstr(cap2[s, z] <= capMax)
                        mod.addConstr(cap2[s, z] >= 0)

                        mod.addConstr(change2[s, z] >= 0)

                        mod.addConstr(change2[s, z] >= (phi2[s, z] - phiX))
                        mod.addConstr(change2[s, z] >= (phiX - phi2[s, z]))

                        mod.addConstr(pDiesel2[s, z] <= pDieselMax)

                        mod.addConstr(pStorageIn2[s, z] <= ubStIn)
                        mod.addConstr(pStorageOut2[s, z] <= ubStOut)
                        mod.addConstr(pGridIn2[s, z] <= ubGrid)

                logging.info("Power balance constraints added successfully")
            except Exception as e:
                logging.error(f"Failed to add power balance constraints: {e}")

            try:
                # build objective (only Power with direct costs)
                o = (sum([cGrid[z] * pGridOut2[s, z] + cDiesel * pDiesel2[s, z] - cGrid[z] * pGridIn2[s, z] +
                          change2[s, z] * cGrid[z] for s in range(scenarios)]) / scenarios)

                logging.info("Built Objective successfully")
            except Exception as e:
                logging.error(f"Failed to build objective: {e}")

            def solve():
                """Optimize VPP planning Model."""
                try:
                    mod.setParam('OutputFlag', 0)
                    mod.optimize()
                    status = mod.status
                    if status == GRB.Status.INF_OR_UNBD or status == GRB.Status.INFEASIBLE \
                            or status == GRB.Status.UNBOUNDED:
                        print('The model cannot be solved because it is infeasible or \
                              unbounded')
                        return False

                    if status != GRB.Status.OPTIMAL:
                        print('Optimization was stopped with status %d' % status)
                        return False

                    logging.info("Model solved successfully with objective value %s", mod.objVal)
                    return True
                except Exception as e:
                    logging.error(f"An error occurred during model optimization: {e}")
                    return False

            try:
                if scenarios == 1:
                    mod.setObjective(c + o / (5 * par))
                elif 1 < scenarios <= 30:
                    mod.setObjective(c + (o - (parNorm / 2)) / par)
                elif 30 < scenarios <= 100:
                    mod.setObjective(c + (o - parNorm) / par)
                else:
                    mod.setObjective(c + (o - (parNorm / 1.5)) / par)

                logging.info("Objective set ended successfully")
            except Exception as e:
                logging.error(f"Failed to set objective: {e}")

            if not solve():
                raise RuntimeError("Model solving failed")

            t = mod.Runtime * 60
            runList.append(t)
            runtime += t

            # extract x values
            a2[i] = pDiesel[i].X
            a4[i] = pStorageIn[i].X
            a5[i] = pStorageOut[i].X
            a3[i] = pRenPV[j][i]
            a6[i] = pGridIn[i].X
            a7[i] = pGridOut[i].X
            a8[i] = cap[i].X
            a1[i] = tilde_cons[i]
            objX[j][i] = mod.objVal
            a9[i] = cGrid[i]
            capX = cap[i].x
            listc.append(capX)
            phiX = phi[i].x

            trace_solutions[j][i] = [mod.objVal, a3[i], a1[i], capX, a2[i], a4[i], a5[i], a6[i], a7[i]]
            objList.append(objX[j][i])

        # computes the memory usage of solve() function, with sampling interval .01 sec
        mem_usage = memory_usage(solve, interval=.01, timeout=1)
        # computes the mean memory usage
        avg_mem_usage = np.mean(mem_usage)
        # get the maximum Resident Set Size (RSS) in kB and converts it in MB
        mem_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.00
        mem_max = mem_max / 1000.00
        # estimates the total memory usage
        mem = np.mean(avg_mem_usage) + 2 * scenarios
        process = psutil.Process(os.getpid())
        test_mem = []

        for i in range(10):
            time.sleep(.01)
            memP = process.memory_percent()
            test_mem.append(round(memP, 2))

        mP = np.mean(test_mem)
        mPmax = (mP * mem_max) / mem
        memPerc.append(round(mP, 2))
        memPercMax.append(round(mPmax, 2))
        memFinal.append(round(avg_mem_usage, 2))
        memMax.append(round(mem_max, 2))

        a10 = shift
        data = np.array([a1, a2, a3, a9, a6, a7, a4, a5, a8, a10])
        for k in range(0, len(objList), 96):
            ob = sum(objList[k:k + 96])
        for k in range(1, len(runList), 96):
            run = sum(runList[k:k + 96])
        runFinal.append(round(run / 3, 2))
        objFinal.append(round(ob, 2))
        test_list = []
        for i in range(10):
            p = psutil.Process(os.getpid())
            p_cpu = p.cpu_percent(interval=0.05)
            test_list.append(round(p_cpu, 3))
        mcpu = np.mean(test_list)
        cpuPercUsage.append(round(mcpu, 3))
        cpuMax.append(round(max(test_list), 3))

        med = np.mean(objFinal)
        var = np.std(objFinal)

        return med, np.mean(runFinal), np.mean(memFinal)


def read_csv_file(file):
    file_path = os.path.join('../data/', file)
    data = pd.read_csv(file_path)
    return data


def read_benchmark_file(file):
    file_path = os.path.join('../data/benchmark/', file)
    columns_to_drop = ['PV(kW)', 'Load(kW)', 'gpuAvg(MB)', 'gpuPeak(MB)', 'gpuEnergy(kW)']
    data = pd.read_csv(file_path)
    data = data.drop(columns=columns_to_drop)
    return data


def display_instances_data(num_rows=2):
    data = read_csv_file('InstancesTest.csv')
    data = data.iloc[:num_rows, 1:]
    data = data.style.set_properties(**{'text-align': 'left'}) \
        .set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'left')]
    }])
    return data


def display_prices_data(num_rows=5):
    data = read_csv_file('PricesGME.csv')
    data = data.rename(columns={'Ora': 'Time', 'Prezzo': 'Price'})
    data = data.iloc[:num_rows, :]
    return data


def display_emissions_data():
    data = read_csv_file('emissions.csv')
    columns = ['run_id', 'gpu_power', 'gpu_energy', 'cloud_provider', 'cloud_region', 'gpu_count', 'gpu_model',
               'on_cloud', 'pue']
    data = data.drop(columns=columns)
    return data


def display_benchmark_data(filename, num_rows=10):
    file_path = os.path.join('benchmark', filename)
    columns_to_drop = ['PV(kW)', 'Load(kW)','gpuAvg(MB)', 'gpuPeak(MB)', 'gpuEnergy(kW)']
    columns_format = {
        'sol(keuro)': '{:.2f}',
        'time(sec)': '{:.2f}',
        'memAvg(MB)': '{:.2f}',
        'memPeak(MB)': '{:.2f}',
        'CO2e(kg)': '{:.2e}',
        'CO2eRate(kg/s)': '{:.2e}',
        'cpuEnergy(kW)': '{:.2e}',
        'ramEnergy(kW)': '{:.2e}',
        'totEnergy(kW)': '{:.2e}',
    }
    data = read_csv_file(file_path)
    data = data.drop(columns=columns_to_drop)
    data = data.iloc[:num_rows, :]
    # format specified columns
    data = data.style.format(columns_format)
    return data


class ConfigDB():
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


class OptimizationRequest():
    """Class that represents and handles an optimization request for HADA. Arguments are checked."""
    def __init__(self,
                 db,
                 algorithm,
                 target,
                 opt_type,
                 robustness_fact,
                 user_constraints,
                 hws_prices):

        if algorithm not in db.get_algorithms():
            raise AttributeError(f'Algorithm {algorithm} not available.')
        self.algorithm = algorithm

        if target not in db.get_targets(algorithm):
            raise AttributeError(f'Target {target} not available for algorithm {algorithm}.')
        self.target = target

        if opt_type not in ['min', 'max']:
            raise AttributeError("Optimization type can only be one of 'min', 'max'.")
        self.opt_type = opt_type

        if not (type(robustness_fact) in [int, float] or robustness_fact is None):
            raise AttributeError('Robustness factor must be numerical or None.')
        self.robustness_fact = robustness_fact

        if not isinstance(user_constraints, UserConstraints):
            raise AttributeError("User constraints must be specified via UserConstraints class.")
        self.user_constraints = user_constraints

        # no user input version; just read from config
        # otherwise we expect prices from user and what's in the configs is only for guidance.
        #self.prices = self.db.get_hw_prices(algorithm)

        if not isinstance(hws_prices, HardwarePrices):
            raise AttributeError("Hardware prices must be specified via HardwarePrices class.")
        self.hws_prices = hws_prices


class UserConstraints():
    """Class that represents an user's constraints to be included in a Request. Arguments are checked."""
    def __init__(self, configdb, algorithm) -> None:

        self.db = configdb

        if algorithm not in self.db.get_algorithms():
            raise AttributeError(f'Algorithm {algorithm} not available.')
        self.algorithm = algorithm

        # target : (type, value)
        self.constraints =  {}

    def add_constraint(self, target, constr_type, value):
        if target not in self.db.get_targets(self.algorithm):
            raise AttributeError(f'Target {target} not available for algorithm {self.algorithm}.')

        if constr_type not in ['eq', 'leq', 'geq']:
            raise AttributeError("Constraint type can only be one of 'eq', 'leq', 'geq'.")

        if type(value) not in [float, int]:
            raise AttributeError("Constraint value must be numerical.")

        self.constraints[target] = (constr_type, value)

    def get_constraints(self):
        return self.constraints


class HardwarePrices():
    """Class that represents the chosen price for each hw platform (algorithm-specific). Arguments are checked."""
    def __init__(self, configdb, algorithm) -> None:

        self.db = configdb

        if algorithm not in self.db.get_algorithms():
            raise AttributeError(f'Algorithm {algorithm} not available.')
        self.algorithm = algorithm

        # hw : price
        # loading default values when specified
        self.__price_per_hw =  {hw: price
                                for hw, price in self.db.get_prices_per_hw(self.algorithm).items()
                                if price}

    def add_hw_price(self, hw, price):
        if hw not in self.db.get_hws(self.algorithm):
            raise AttributeError(f'Hardware platform {hw} not available for algorithm {self.algorithm}.')

        # ignore if price is None
        if price is None:
            return

        if type(price) not in [float, int]:
            raise AttributeError("Price must be numerical.")

        self.__price_per_hw[hw] = price

    def get_prices_per_hw(self):
        # checking that all prices for the algorithms are specified
        hws = self.db.get_hws(self.algorithm)

        if not set(hws) == set(self.__price_per_hw.keys()):
            raise AttributeError("Prices for all hardware platforms related to the algorithm must be specified when the target is 'price' or 'price' is constrainted.")
        return self.__price_per_hw


class OptimizationSolution():
    '''Class containing a solution produced by HADA.'''
    def __init__(self, chosen_hw, hyperparams_values, targets_values):
        self.chosen_hw = chosen_hw
        self.hyperparams_values = hyperparams_values
        self.targets_values = targets_values

    def __str__(self):
        return f'chosen hw: {self.chosen_hw}; hyperparams values: {self.hyperparams_values}; targets values: {self.targets_values}'


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


    @abstractmethod
    def get_dataset(self, algorithm, hw) -> pd.DataFrame:
        """Returns the dataset (Pandas DataFrame) relative to the (algorithm, hw), if present."""
        pass

    def _check_dataset_consistency(self, df, algorithm, hw):
        """Checking the columns are the expected ones and that they are numericals."""
        hyperparams = self.db.get_hyperparams(algorithm)
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


class MLModels():
    def __init__(self, db, datasets, models_path):
        """Handles all operations on ML models.

        Args:
            db (ConfigDB): ConfigDB instance.
            datasets (Datasets): Datasets instance.
            models_path (str): local path where the models are stored.
        """
        self.db = db
        self.models_path = models_path
        self.datasets = datasets

        # tracking state about (algorithm, hw, target) that are currently being trained
        self.ongoing_training = Manager().dict()

    def __get_model_path(self, algorithm, hw, target):
        return os.path.join(self.models_path, f'{algorithm}_{hw}_{target}_DecisionTree_10')

    def get_model(self, algorithm, hw, target):
        """Returns the model (Decision).

        Args:
            algorithm (str): algorithm id.
            hw (str): hardware platform id
            target (str): target id.

        Raises:
            Exception: if model is not found and is already being trained.

        Returns:
            sklearn.tree.DecisionTreeRegressor: DT model.
        """
        model_path = self.__get_model_path(algorithm, hw, target)

        if not os.path.exists(model_path):
            if (algorithm, hw, target) in self.ongoing_training:
                raise Exception(f'Model for ({algorithm}, {hw}, {target}) training is ongoing. Come back later.')
            else:
                # launching training in background
                dataset = self.datasets.get_dataset(algorithm, hw)
                self.ongoing_training[(algorithm, hw, target)] = True
                p = Process(target=self.__run_training, args=(algorithm,
                                                              hw,
                                                              target,
                                                              dataset))
                p.start()
                # raise FileNotFoundError(f'Model for ({algorithm}, {hw}, {target}) does not exist. Training started. Come back later.')
                # without the Exception, nothing is shown in the GUI, but multiple models can be trained in a single
                # request, while still keeping all the training part incapsulated in "get_model"
                print(f'Model for ({algorithm}, {hw}, {target}) does not exist. Training started.')
                p.join()
                del self.ongoing_training[(algorithm, hw, target)]
                print(f'Finished training model for ({algorithm}, {hw}, {target}).')

        # model exists, load it
        model = pickle.load(open(model_path, 'rb'))
        return model

    def __run_training(self, algorithm, hw, target, dataset):
        """
        Trains a Decision Tree and stores it with pickle.

        Args:
            algorithm (str): algorithm id.
            hw (str): hardware platform id.
            target (str): target id.
            dataset (pd.DataFrame): training dataset.

        """
        # s = time.time()
        model_path = self.__get_model_path(algorithm, hw, target)

        # filtering dataset for the specific hyperparams and target
        hyperparams = self.db.get_hyperparams(algorithm)
        X = dataset[hyperparams].values
        y = dataset[[target]].values

        # training the DT
        dt = DecisionTreeRegressor(max_depth=10, random_state=42)
        dt.fit(X, y)

        # storing the DT
        pickle.dump(dt, open(model_path, 'wb'))

        # print(self.ongoing_training)
        # print(f'Done in {time.time()-s}')
        # del self.ongoing_training[(algorithm, hw, target)]
        # print(self.ongoing_training)


def HADA(db: ConfigDB, request, models, var_bounds, robust_coeff):
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
