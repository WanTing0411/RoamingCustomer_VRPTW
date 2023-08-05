# Roaming Customer VRP Time Window Problem

This repository contains a solution to the VRPTW using a linear programming(Gurobi) approach. The problem involves efficiently routing a fleet of vehicles to serve a set of customers with time window constraints, optimizing for the total cost.

## Introduction

The Roaming Customer VRP Time Window Problem is a classic optimization problem that arises in logistics and transportation management. The objective is to find the optimal routes for a fleet of vehicles to deliver goods to a set of customers within their specified time windows while minimizing the total cost.

## Features

- **Problem Definition**: The RCVRPTW involves a set of customers with several potential specific time windows and demands. Vehicles with limited capacity aim to serve these customers efficiently.

- **Input Format**: The input data consists of customer coordinates, time window intervals, demand values, vehicle capacity, depot opening hours, cost of vehicles, and other relevant parameters.

- **Algorithm/Approach**: Our solution employs the Gurobi method, which iteratively evolves a set of solutions to find a near-optimal solution for a small sample RCVRPTW.
- 
- **Output Format**: The output includes the routes for each vehicle, the order of customer visits, the total distance traveled, and the total cost of ownership.

- **Usage of Libraries/Tools**: We utilize the Python `numpy` library for array operations and  `Gurobi` for optimizing routes and `matplotlib` for visualizing the routes.

## Examples

### Example Input

Vehicle 1: Depot -> Customer 3 -> Customer 1 -> Depot (Distance: 120)
Vehicle 2: Depot -> Customer 2 -> Customer 5 -> Customer 4 -> Depot (Distance: 180)
