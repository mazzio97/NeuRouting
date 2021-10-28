import os
from typing import List

import numpy as np
from more_itertools import split_after

from instances import VRPInstance


GRID_DIM = 100000


def read_vrp(filepath, grid_dim: int = GRID_DIM) -> VRPInstance:
    with open(filepath, "r") as f:
        lines = [ll.strip() for ll in f]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                size = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + size], dtype=int)
                i = i + size
            elif line.startswith('DEMAND_SECTION'):
                demands = np.loadtxt(lines[i + 1:i + 1 + size], dtype=int)
                i = i + size
            i += 1

    def norm(coord: int) -> float:
        return float(coord) / grid_dim

    return VRPInstance(
        depot=(norm(locations[0][1]), norm(locations[0][2])),
        customers=[(norm(loc[1]), norm(loc[2])) for loc in locations[1:]],
        demands=[d[1] for d in demands[1:]],
        capacity=capacity
    )


def write_vrp(instance: VRPInstance, filepath: str, grid_dim=GRID_DIM):
    with open(filepath, 'w') as f:
        f.write("\n".join([
            f"{key} : {value}"
            for key, value in (
                ("NAME", os.path.splitext(filepath)[0].split('/')[-1]),
                ("TYPE", "CVRP"),
                ("DIMENSION", instance.n_customers + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", instance.capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate([instance.depot] + instance.customers):
            x, y = int(x * grid_dim + 0.5), int(y * grid_dim + 0.5)
            f.write(f"{i + 1}\t{x}\t{y}\n")
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([f"{i + 1}\t{d}" for i, d in enumerate([0] + instance.demands)]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def read_solution(filename: str, n: int) -> List[List[int]]:
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    tour = tour[1:].tolist() + [0]
    return list([0] + t for t in split_after(tour, lambda x: x == 0))
