#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
@author: allegradefilippo
"""

from gurobipy import Model
from gurobipy import GRB
import numpy as np
from memory_profiler import profile
from memory_profiler import memory_usage

import pandas as pd
import psutil
import os
import time
import resource
import gc
import logging
import csv

from codecarbon import EmissionsTracker
from src import gpuMonitor


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



def onlineAnt(ns, mr, file_path):
    """
    Runs the Online Anticipate algorithm on a given instance.

    :param ns: Number of scenarios.
    :param mr: Instance id.
    :param file_path: path of .csv file containing the instances.
    """

    # timestamps n instances m scenarios ns
    n = 96
    par = 100
    parNorm = ns
    m = 1

    # to save it in emissions.csv
    project_name = f"anticipate-ins-{mr}-ns-{ns}"
    print("Started online ant")
    print(project_name)

    # Codecarbon emission tracker
    tracker = EmissionsTracker(project_name=project_name)

    # file paths
    input_files_path = os.path.join(os.getcwd(), 'inputDataFiles')
    utils_path = os.path.join(os.getcwd(), 'utils')
    output_path = os.path.join(os.getcwd(), 'out')

    # price data from GME
    prices_path = os.path.join(input_files_path, 'PricesGME.csv')
    cGrid = parse_data_gme(prices_path)
    cGridSt = np.mean(cGrid)

    # capacities and prices
    listc = []
    objX = np.zeros((m, n))
    objTot = [None] * m
    avg_mem_usage = [] * m
    gpu_usage = [] * m

    listc, objList, runList, objFinal, runFinal, memFinal, cpuPercUsage, mem, cpuMax, memMax, memPerc, memPercMax, \
        gpuFinal, gpuMax = \
        [[] for _ in range(14)]

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

    instances = pd.read_csv(file_path)

    # instances pv
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: entry[1:-1].split())
    instances['PV(kW)'] = instances['PV(kW)'].map(lambda entry: list(np.float_(entry)))

    pRenPV = [instances['PV(kW)'][mr] for i in range(m)]
    np.asarray(pRenPV)

    # instances load
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: entry[1:-1].split())
    instances['Load(kW)'] = instances['Load(kW)'].map(lambda entry: list(np.float_(entry)))
    totCons = [instances['Load(kW)'][mr] for i in range(m)]
    np.asarray(totCons)

    shift = np.load(os.path.join(utils_path, 'realizationsShift.npy'))
    capX1 = 200

    for j in range(m):
        gc.collect(2)
        gc.set_debug(gc.DEBUG_SAVEALL)
        #    print ('Realization',j,'resolution:')

        print("start emission tracker")
        # start emission tracker
        tracker.start()

        # print(f"for {j} in {range(m)}")

        for i in range(n):
            # n2 = remaining step for scenarios
            n2 = n - i

            # print(f"for {i} in {range(n)}")

            # ns = scenarios
            cap2 = np.zeros((ns, n2))
            pDiesel2 = np.zeros((ns, n2))
            pStorageIn2 = np.zeros((ns, n2))
            pStorageOut2 = np.zeros((ns, n2))
            pGridIn2 = np.zeros((ns, n2))
            pGridOut2 = np.zeros((ns, n2))
            runtime = 0

            tilde_cons_scen = np.zeros((ns, n2))
            pRenPV_scen = np.zeros((ns, n2))

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
            #        print tilde_cons

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
                pDiesel2 = mod.addVars(ns, n2, vtype=GRB.CONTINUOUS, name="pDiesel_")
                pStorageIn2 = mod.addVars(ns, n2, vtype=GRB.CONTINUOUS, name="pStorageIn_")
                pStorageOut2 = mod.addVars(ns, n2, vtype=GRB.CONTINUOUS, name="pStorageOut_")
                pGridIn2 = mod.addVars(ns, n2, vtype=GRB.CONTINUOUS, name="pGrid_")
                pGridOut2 = mod.addVars(ns, n2, vtype=GRB.CONTINUOUS, name="pGrid_")
                cap2 = mod.addVars(ns, n2, vtype=GRB.CONTINUOUS, name="cap_")
                change2 = mod.addVars(ns, n2, vtype=GRB.INTEGER, name="change")
                phi2 = mod.addVars(ns, n2, vtype=GRB.BINARY, name="phi")
                notphi2 = mod.addVars(ns, n2, vtype=GRB.BINARY, name="notphi")

                logging.info("Second stage variables added successfully")
            except Exception as e:
                logging.error(f"Failed to add Second stage variables: {e}")

            try:

                # Define Scenarios
                all_load_scen = np.load(os.path.join(utils_path, 'outfileLoad.npy'))
                all_PV_scen = np.load(os.path.join(utils_path, 'outfilePV.npy'))

                load_scen = np.zeros((ns, n))
                PV_scen = np.zeros((ns, n))

                np.random.seed(3)
                for s in range(ns):
                    load_scen[s] = np.random.choice(all_load_scen[s], n)

                for s in range(ns):
                    PV_scen[s] = np.random.choice(all_PV_scen[s], n)

                for s in range(ns):
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
                                for s in range(ns) for z in range(n2)), "Power balance")

                for s in range(ns):
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
                          change2[s, z] * cGrid[z] for s in range(ns)]) / ns)

                logging.info("Built Objective successfully")
            except Exception as e:
                logging.error(f"Failed to build objective: {e}")

            fp = open(os.path.join(output_path, 'memory_profiler.log'), 'w+')

            @profile(stream=fp)
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
                if ns == 1:
                    mod.setObjective(c + o / (5 * par))
                elif 1 < ns <= 30:
                    mod.setObjective(c + (o - (parNorm / 2)) / par)
                elif 30 < ns <= 100:
                    mod.setObjective(c + (o - parNorm) / par)
                else:
                    mod.setObjective(c + (o - (parNorm / 1.5)) / par)

                logging.info("Objective set ended successfully")
            except Exception as e:
                logging.error(f"Failed to set objective: {e}")

            # start gpu monitoring if a gpu is available

            if gpuMonitor.is_gpu_available():
                gpuMonitor.start_monitoring(interval=0.01)

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

        # stop emission tracker
        tracker.stop()

        #    time.sleep(0.005)
        try:
            # computes the memory usage of solve() function, with sampling interval .01 sec
            mem_usage = memory_usage(solve, interval=.01, timeout=1)
            # computes the mean memory usage
            avg_mem_usage = np.mean(mem_usage)
            # print(f"average memory usage = {avg_mem_usage}")
            # get the maximum Resident Set Size (RSS) in kB and converts it in MB
            mem_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.00
            mem_max = mem_max / 1000.00
            # print(f"maximum mem usage = {mem_max}")
            # estimates the total memory usage
            mem = np.mean(avg_mem_usage) + 2 * ns
            # print(f"total memory estimate, taking ns into account = {mem}")
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

            # check and process gpu usage if available
            if gpuMonitor.is_gpu_available():
                gpu_usage = gpuMonitor.gpu_stats # get the collected gpu usage stats
                if gpu_usage:
                    # Calculate average and max GPU usage
                    average_gpu_usage = np.mean(gpu_usage)
                    max_gpu_usage = np.max(gpu_usage)
                    gpuFinal.append(average_gpu_usage)
                    gpuMax.append(max_gpu_usage)
                else:
                    # Handle the case where gpu_usage is empty
                    print("Warning: GPU usage data is empty.")
                    gpuFinal.append(-1)  # or another appropriate value
                    gpuMax.append(-1)
            else:
                # No GPU
                gpuFinal.append(-1)
                gpuMax.append(-1)

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

            print(f"The solution cost (in keuro) is: {med:.2f}")
            print(f"The runtime (in sec) is: {np.mean(runFinal):.2f}")
            print(f"Avg memory used (in MB) is: {np.mean(memFinal):.2f}")
            print(f"Avg GPU usage (in MB) is: {np.mean(gpuFinal):.2f}")

            # save .npy files ("utils" directory)
            np.save(os.path.join(utils_path, 'obj.npy'), np.asarray(objFinal))
            np.save(os.path.join(utils_path, 'run.npy'), np.asarray(runFinal))
            np.save(os.path.join(utils_path, 'mem_avg.npy'), np.asarray(memFinal))
            np.save(os.path.join(utils_path, 'mem_max.npy'), np.asarray(memMax))
            np.save(os.path.join(utils_path, 'gpu_avg.npy'), np.asarray(gpuFinal))
            np.save(os.path.join(utils_path, 'gpu_max.npy'), np.asarray(gpuMax))

        finally:
            # ensure that gpu monitoring stops even if an error occurs
            if gpuMonitor.is_gpu_available():
                gpuMonitor.stop_monitoring_gpu()

