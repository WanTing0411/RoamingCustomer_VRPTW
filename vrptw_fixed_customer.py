import datetime
import csv
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import gurobipy
from gurobipy import quicksum
from math import radians, cos, sin, asin, sqrt, pi
from datetime import datetime, date
import matplotlib.pyplot as plt
import psutil
import os


class Node:
    def __init__(self, log, lat, tw_start, tw_end, service_time, demand):
        self.log = log
        self.lat = lat
        self.tw_start = tw_start
        self.tw_end = tw_end
        self.service_time = service_time
        self.demand = demand

    def print(self):
        print("log:", self.log, " lat:", self.lat, " strat_time:", self.tw_start, " end_time:", self.tw_end,
              " service_time:", self.service_time, "mins")


class Customer:
    def __init__(self, name, demand):
        self.name = name
        self.demand = demand
        self.nodes = []


class Vehicle:
    def __init__(self, vehicle_id, fixed_cost, capacity, duration, avg_velocity):
        self.vehicle_id = vehicle_id
        self.fixed_cost = fixed_cost
        self.capacity = capacity
        self.duration = duration
        self.avg_velocity = avg_velocity


class Depot:
    def __init__(self, depot_id, depot_log, depot_lat, open_time, close_time):
        self.depot_id = depot_id
        self.depot_log = depot_log
        self.depot_lat = depot_lat
        self.open_time = open_time
        self.close_time = close_time


# read customer data
random_customer = pd.read_csv('vrptw_fixed_customer_coordinates(real data 15c).csv')

random_customer = random_customer.reset_index()
# read vehicle data
vehicle_data = pd.read_csv('vehicle_data_k.csv')
vehicle_data = vehicle_data.reset_index()
# read depot data
depot_data = pd.read_csv('depot_data_k.csv')
depot_data = depot_data.reset_index()

customerList: List[Customer]
nameList: List[str]
## dictionary <customerID, customerInstance>
customerDict: Dict[str, Customer] = dict()

vehicleList: List[Vehicle]
## dictionary <vehicleID, vehicleInstance>
vehicleDict: Dict[str, Vehicle] = dict()

depotList: List[Depot]
## dictionary <depotID, depotInstance>
depotDict: Dict[str, Depot] = dict()
print("--------------------Data from csv--------------------")
for index, row in random_customer.iterrows():
    if row["CustomerID"] not in list(customerDict):  # add customer in Dict without repeating
        customerDict.update({row["CustomerID"]: Customer(row["CustomerID"], row["Demand_kg"])})
    customerDict[row["CustomerID"]].nodes.append(
        Node(row["Customer_Longtitude"], row["Customer_Latitude"], row["tw_start"], row["tw_end"], row["service_time"],
             row["Demand_kg"]))
    # add nodes list in customer

for customer in list(customerDict):
    print("Customer_ID:", customerDict[customer].name, ",Total Demand:", customerDict[customer].demand)
    for node in customerDict[customer].nodes:
        node.print()

print()

for index, row in vehicle_data.iterrows():
    vehicleDict.update(
        {row["Vehicle_ID"]: Vehicle(row["Vehicle_ID"], row["Vehicle_Fixed_Cost"], row["Vehicle_Capacity"],
                                    row["Max_duration"], row["AvgVelocity_km/h"])})

for vehicle in list(vehicleDict):
    print("VehicleID:", vehicleDict[vehicle].vehicle_id, " FixedCost:", vehicleDict[vehicle].fixed_cost, " Capacity:",
          vehicleDict[vehicle].capacity, "MaxDuration:", vehicleDict[vehicle].duration, "AvgVelocity(km/h):",
          vehicleDict[vehicle].avg_velocity)

print()
for index, row in depot_data.iterrows():
    depotDict.update(
        {row["Depot_ID"]: Depot(row["Depot_ID"], row["Depot_Long"], row["Depot_Lat"], row["Open_time"],
                                row["Close_time"])})

for depot in list(depotDict):
    print("DepotID:", depotDict[depot].depot_id, " Depot_Long:", depotDict[depot].depot_log, " Depot_Lat:",
          depotDict[depot].depot_lat, " Open time:", depotDict[depot].open_time, " Close time:",
          depotDict[depot].close_time)
print()

# Gurobi
# Noation setup
C = [i for i in list(customerDict)]  # set of customers

N = []
N_withCustomer = []  # 1)set of nodes
for i in list(customerDict):
    for j in customerDict[i].nodes:
        if (j.log, j.lat, customerDict[i].name, j.tw_start) not in N_withCustomer:  # prevent repeat node
            N_withCustomer.append((j.log, j.lat, customerDict[i].name, j.tw_start))
            N.append((j.log, j.lat))

N0 = []  # set of nodes including the depot
N0.append((depotDict[depot].depot_log, depotDict[depot].depot_lat))
for i in N:
    N0.append(i)

Nr = []  # customer potential node
Nr_withTimeDiffer = []
for r in list(customerDict):
    Cn = []  # customer potential node
    for i in customerDict[r].nodes:
        if (i.log, i.lat, customerDict[r].name, i.tw_start,
            i.tw_end) not in Nr_withTimeDiffer:  # prevent repeat node in each customer data
            Nr_withTimeDiffer.append((i.log, i.lat, customerDict[r].name, i.tw_start, i.tw_end))
            Cn.append((i.log, i.lat))
    Nr.append(Cn)

V = [i for i in list(vehicleDict)]  # set of vehicles

R = []  # set request
for i in list(customerDict):
    R.append(customerDict[i].demand)

Q = vehicleDict[vehicle].capacity  # capacity of vehicle
L = vehicleDict[vehicle].duration  # max duration for driver

init_time = "0:00"
init_time_struct = datetime.strptime(init_time, "%H:%M")
time_struct = datetime.strptime(depotDict[depot].open_time, "%H:%M")  # depot closing time
minuteOpen = (time_struct - init_time_struct).seconds / 60  # depot closing time transfer to minutes

init_time = "0:00"
init_time_struct = datetime.strptime(init_time, "%H:%M")
time_struct = datetime.strptime(depotDict[depot].close_time, "%H:%M")  # depot closing time
minuteclose = (time_struct - init_time_struct).seconds / 60
O = minuteOpen
E = minuteclose  # depot closing time transfer to minutes

# Parameter
ai = []  # openining tw for customer arrvie node
init_time = "0:00"
init_time_struct = datetime.strptime(init_time, "%H:%M")
temporary_str = []
for i in list(customerDict):
    time_str_node = []  # prevent repeat nodes (e.g. c2)
    for j in customerDict[i].nodes:  # customer potential node
        if (j.tw_start not in time_str_node):
            time_str_node.append(j.tw_start)
    temporary_str.append(time_str_node)
for i in temporary_str:
    for j in i:
        each_time_struct = datetime.strptime(j, "%H:%M")
        minutes = (
                          each_time_struct - init_time_struct).seconds / 60  # 1) in order to calclulate, need to transfer excel time format to number 2) unit: minute
        ai.append(minutes)

bi = []  # closing tw for customer leave node
temporary_end = []
for i in list(customerDict):
    time_end_node = []
    for j in customerDict[i].nodes:  # customer potential node
        if (j.tw_end not in time_end_node):
            time_end_node.append(j.tw_end)
    temporary_end.append(time_end_node)
for i in temporary_end:
    for j in i:
        each_time_struct = datetime.strptime(j, "%H:%M")
        minutes = (
                          each_time_struct - init_time_struct).seconds / 60  # 1) in order to calclulate, need to transfer excel time format to number 2) unit: minute
        bi.append(minutes)

sigma_i = 1  # assume each node service time=1


def distance(p1, p2):
    lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2  # transfer long & lat to distance
    c = 2 * asin(sqrt(a))
    r = 6371  # r:represents the radius of the earth
    dis = round((c * r * 1000) / 1000, 2)
    return dis


index = 0
Cij = []  # travel cost between all pairs of nodes
for i in N0:
    travel_cost = []
    for j in range(0, len(N0)):
        time_travel = distance(N0[index], N0[j]) / (vehicleDict[vehicle].avg_velocity) * 60  # km/hr transfer min
        travel_cost.append(time_travel)
    index += 1
    Cij.append(travel_cost)

qi = []  # the potential demand in node (depend on whether this potentila node will be loaded on the final route ->need to Xijv to decide
qi.append(0)  # the request of depot
for i in list(customerDict):
    for j in customerDict[i].nodes:  # customer potential node
        qi.append(j.demand)

fv = vehicleDict[vehicle].fixed_cost

print("--------------------Decision Variable--------------------")
m = gurobipy.Model('VRPRD LP Model')

pairs = [(i, j, v) for i in range(len(N0)) for j in range(len(N0)) for v in range(len(V))]
X = m.addVars(pairs, vtype=gurobipy.GRB.BINARY, name='var_X_ijv')
Y = m.addVars(V, vtype=gurobipy.GRB.BINARY, name='var_Y_v')
ti = m.addVars(range(len(N0)), lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='var_t_i')
tr = m.addVars(range(len(R)), lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='var_t_r')
t_alpha = m.addVars(V, lb=0,ub=1080, vtype=gurobipy.GRB.CONTINUOUS, name='var_t_alpha')
t_delta = m.addVars(V, lb=0, vtype=gurobipy.GRB.CONTINUOUS, name='var_t_delta')
S = m.addVars(range(len(N0)), lb=0, ub=Q, vtype=gurobipy.GRB.CONTINUOUS, name='var_s_i')
m.update()

print("--------------------Objective function--------------------")
m.modelSense = gurobipy.GRB.MINIMIZE
objective = gurobipy.LinExpr()
CH_cost = 0.25  # 12 euro for 1 hr->0.25 euro for mins
tv_cost = 0.024 #1.39 euro for 1 hr->0.024 euro for mins
sum_route_cost = 0
sum_vehicle_cost = 0
for i in range(len(N0)):
    for j in range(len(N0)):
        for v in range(len(V)):
            sum_route_cost += X[i, j, v] * (
                    (Cij[i][j] * tv_cost) + ((t_alpha.values()[v] - t_delta.values()[v]) * CH_cost))

for v in range(len(V)):
    sum_vehicle_cost += Y.values()[v] * fv
objective = sum_route_cost + sum_vehicle_cost
m.setObjective(objective)

print("--------------------Constraint--------------------")

for v in range(len(V)):
    for j in range(len(N0)):
        ct_1_lhs = gurobipy.LinExpr()
        ct_1_rhs = gurobipy.LinExpr()
        for i in range(len(N0)):
            if i == j:
                continue
            ct_1_lhs += X[i, j, v]  # sum of first nodes range
            ct_1_rhs += X[j, i, v]  # sum of second nodes range
        m.addConstr(ct_1_lhs == ct_1_rhs, name='constraint_1')

for v in range(len(V)):
    for i in range(len(N)):
        ct_2_lhs = gurobipy.LinExpr()
        ct_2_lhs = X[i + 1, i + 1, v]  # because DV Xijv i&j in N0
        m.addConstr(ct_2_lhs == 0, name='constraint_2')

for v in range(len(V)):
    ct_3_1_lhs = gurobipy.LinExpr()
    ct_3_2_lhs = gurobipy.LinExpr()
    for i in range(len(N0)):
        ct_3_1_lhs += X[i, 0, v]
    m.addConstr(ct_3_1_lhs == 1, name='constraint_3_1')
    for j in range(len(N0)):
        ct_3_2_lhs += X[0, j, v]
    m.addConstr(ct_3_2_lhs == 1, name='constraint_3_2')

index = 0
for r in range(len(Nr)):
    ct_4_lhs = gurobipy.LinExpr()
    for i in range(len(Nr[r])):
        for v in range(len(V)):
            for j in range(len(N0)):
                ct_4_lhs += X[j, index + 1, v]
        index += 1
    m.addConstr(ct_4_lhs == 1, name='constraint_4')

M = 9000
ct_5_lhs = gurobipy.LinExpr()  # i benlong N0 should change N because service time will add in depot
ct_5_rhs = gurobipy.LinExpr()
for v in range(len(V)):
    for i in range(len(N0)):
        for j in range(len(N)):
            if i == 0:
                ct_5_lhs = Cij[i][j + 1] + ti.values()[i]
                ct_5_rhs = ti.values()[j + 1] + (M * (1 - X[i, j + 1, v]))
            else:
                ct_5_lhs = Cij[i][j + 1] + sigma_i + ti.values()[
                    i]  # ti is dictionary #ti.values() is a array, so I use [i] to find the index i element
                ct_5_rhs = ti.values()[j + 1] + (M * (1 - X[i, j + 1, v]))
            m.addConstr(ct_5_lhs <= ct_5_rhs, name='constraint_5')

for i in range(len(N)):
    ct_6_lhs_1 = gurobipy.LinExpr()
    ct_6_rhs_1 = gurobipy.LinExpr()
    ct_6_rhs_2 = gurobipy.LinExpr()
    sum = gurobipy.LinExpr()
    for j in range(len(N0)):
        for v in range(len(V)):
            if j == i + 1:
                continue
            ct_6_lhs_1 += ai[i] * X[i + 1, j, v]
            ct_6_rhs_1 = ti.values()[i + 1]
            ct_6_rhs_2 += bi[i] * X[i + 1, j, v]
    m.addConstr(ct_6_lhs_1 <= ct_6_rhs_1, name='constraint_6_1')
    m.addConstr(ct_6_rhs_1 <= ct_6_rhs_2, name='constraint_6_2')

ct_7_lhs = gurobipy.LinExpr()
index = 0
for r in range(len(R)):
    ct_7_lhs = tr.values()[r]  # the request of one customer
    ct_7_rhs = gurobipy.LinExpr()
    for j in range(len(Nr[r])):  # run the list of Nr:customer
        index += 1  # in order to skip ti[0]-> the depot node
        ct_7_rhs += ti.values()[index]
    m.addConstr(ct_7_lhs == ct_7_rhs, name='constraint_7')

ct_8_lhs = gurobipy.LinExpr()
ct_8_rhs = gurobipy.LinExpr()
for v in range(len(V)):
    for i in range(len(N)):
        ct_8_lhs += X[0, i + 1, v]
    ct_8_rhs += Y.values()[v]
m.addConstr(ct_8_lhs <= ct_8_rhs, name='constraint_8')

ct_9_lhs = gurobipy.LinExpr()
ct_9_rhs = gurobipy.LinExpr()
for v in range(len(V)):
    for i in range(len(N)):
        for j in range(len(N0)):
            ct_9_lhs = S.values()[j] - qi[i + 1] * X[i + 1, j, v]
            ct_9_rhs = S.values()[i + 1] - Q * (1 - X[i + 1, j, v])
            m.addConstr(ct_9_lhs >= ct_9_rhs, name='constraint_9')

ct_10_lhs_1 = gurobipy.LinExpr()
ct_10_rhs_1 = gurobipy.LinExpr()
ct_10_lhs_2 = gurobipy.LinExpr()
ct_10_rhs_2 = gurobipy.LinExpr()
for v in range(len(V)):
    for i in range(len(N)):  # N should change N0
        ct_10_lhs_1 = t_delta.values()[v] + Cij[0][i + 1]
        ct_10_rhs_1 = ti.values()[i + 1] + M * (1 - X[0, i + 1, v])
        ct_10_lhs_2 = ti.values()[i + 1] + sigma_i + Cij[i + 1][0]
        ct_10_rhs_2 = t_alpha.values()[v] + M * (1 - X[i + 1, 0, v])
        m.addConstr(ct_10_lhs_1 <= ct_10_rhs_1, name='constraint_10_1')
        m.addConstr(ct_10_lhs_2 <= ct_10_rhs_2, name='constraint_10_2')

ct_11_lhs = gurobipy.LinExpr()
ct_11_rhs = gurobipy.LinExpr()
for v in range(len(V)):
    ct_11_lhs = t_delta.values()[v]
    ct_11_rhs = t_alpha.values()[v]
    m.addConstr(ct_11_lhs <= ct_11_rhs, name='constraint_11')


ct_12_lhs = gurobipy.LinExpr()
for v in range(len(V)):
    ct_12_lhs = t_alpha.values()[v] - t_delta.values()[v]
    m.addConstr(ct_12_lhs <= L, name='constraint_12')


ct_13_lhs_1 = gurobipy.LinExpr()
ct_13_lhs_2 = gurobipy.LinExpr()
ct_13_rhs = gurobipy.LinExpr()
for v in range(len(V)):
    ct_13_lhs_1 = t_delta.values()[v]  # departure time
    ct_13_rhs = M * (1 - X[0, 0, v])
    ct_13_lhs_2 = t_alpha.values()[v]  # arrival time
    m.addConstr(ct_13_lhs_1 <= ct_13_rhs, name='constraint_13_1')
    m.addConstr(ct_13_lhs_1 >= (-ct_13_rhs), name='constraint_13_2')
    m.addConstr(ct_13_lhs_2 <= ct_13_rhs, name='constraint_13_3')
    m.addConstr(ct_13_lhs_2 >= (-ct_13_rhs), name='constraint_13_4')

m.Params.MIPGAP = 0.01
m.Params.TimeLimit = 300 #time limit:5mins
m.optimize()

print("The optimal obj. value is:", m.ObjVal)
index = 0;
for v in range(len(V)):
    print("Route", index + 1, ":")
    index += 1
    for i in range(len(N0)):
        for j in range(len(N0)):
            if X[i, j, v].X == 1:
                print(f"The value of  x({i}_{j}_{v + 1}):", X[i, j, v].X, f"residual capacity is ", S.values()[j].X)

print()

index = 0
for v in range(len(V)):
    if Y.values()[v].X == 1:
        index += 1
print(f"Total number of vehicles used:", index)

print()
index = 0
for r in range(len(R)):
    print()
    print(f"Customer {r + 1}:")
    for j in range(len(Nr[r])):  # run the list of Nr:customer
        index += 1  # in order to skip ti[0]-> the depot node
        ti.values()[index] += ti.values()[index]
        print(f"The delivery time in customer{r + 1}_node{j + 1} of request,", ti.values()[index].X)

print()
for v in range(len(V)):
    print(f"The arrval time at depot of vehicle {v + 1} is:", t_alpha.values()[v].X)

print()
for v in range(len(V)):
    if t_delta.values()[v].X >=0:
        print(f"The departure time at depot of vehicle {v + 1} is:", t_delta.values()[v].X)


print()

x = sin(pi / 2 - N0[0][1]) * cos(N0[0][0])
y = sin(pi / 2 - N0[0][1]) * sin(N0[0][0])
print(x, y)

lon = []
lon.append((depotDict[depot].depot_log))
lat = []
lat.append((depotDict[depot].depot_lat))
try1 = []  # 1)set of nodes
for i in list(customerDict):
    for j in customerDict[i].nodes:
        if (j.log, j.lat, customerDict[i].name, j.tw_start) not in try1:  # prevent repeat node
            try1.append((j.log, j.lat, customerDict[i].name, j.tw_start))
            lon.append((j.log))
            lat.append((j.lat))

plt.plot(N0[0][0], N0[0][1], c='r', marker='s')  # print depot(x,y)
plt.scatter(lon[0:], lat[0:], c='b')

for i in range(len(N0)):
    for j in range(len(N0)):
        if X[i, j, 0].X == 1:
            plt.plot([lon[i], lon[j]], [lat[i], lat[j]], c='g')
        elif X[i, j, 1].X == 1:
            plt.plot([lon[i], lon[j]], [lat[i], lat[j]], c='y')
        elif X[i, j, 2].X == 1:
            plt.plot([lon[i], lon[j]], [lat[i], lat[j]], c='r')
        elif X[i, j, 3].X == 1:
            plt.plot([lon[i], lon[j]], [lat[i], lat[j]], c='m')
        elif X[i, j, 4].X == 1:
            plt.plot([lon[i], lon[j]], [lat[i], lat[j]], c='pink')
        elif X[i, j, 5].X == 1:
            plt.plot([lon[i], lon[j]], [lat[i], lat[j]], c='c')

plt.show()
