import os
from typing import List

import numpy as np
from more_itertools import split_after

from nlns.instances import VRPInstance, VRPSolution


GRID_DIM = 100000


def _norm(coord: int, grid_dim=GRID_DIM) -> float:
    """Normalize coordinate given a grid size."""
    return float(coord) / grid_dim


def read_vrp_str(vrp_string: str, grid_dim: int = GRID_DIM) -> VRPInstance:
    """Read a VRP instance from string.

    Args:
        vrp_string: The input string.
        grid_dim: Dimension of the grid. The input coordinates are
            expected to be integers, while neurouting internal
            representation is constrained in a unit square. The grid
            dimension is used to normalize coordinates accordingly.

    Returns:
        A VRP instance.
    """
    lines = tuple(map(str.strip, vrp_string.splitlines()))
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

    return VRPInstance(
        depot=(_norm(locations[0][1]), _norm(locations[0][2])),
        customers=[(_norm(loc[1]), _norm(loc[2])) for loc in locations[1:]],
        demands=[d[1] for d in demands[1:]],
        capacity=capacity
    )


def read_vrp(filepath='', file=None, grid_dim: int = GRID_DIM) -> VRPInstance:
    """Read a VRP instance from file.

    Args:
        filepath: Filename or path to read the instance from.
        file: A filelike object to read from. If given, overrides
            ``filepath``
        grid_dim: See :func:`read_vrp_str`.

    Returns:
        A VRP instance.
    """
    if file is None:
        with open(filepath) as fin:
            vrp_string = fin.read()
    else:
        vrp_string = file.read()

    return read_vrp_str(vrp_string, grid_dim)


def write_vrp_str(instance: VRPInstance, name: str, grid_dim=GRID_DIM) -> str:
    """Write a VRP instance to string.

    Args:
        instance: The input instance.
        name: Arbitrary string name for the instance.
        grid_dim: See :func:`vrp_read_str`.

    Returns:
        A string representing the instance.
    """
    lines = []
    for key, value in (
                ('NAME', name),
                ('TYPE', 'CVRP'),
                ('DIMENSION', instance.n_customers + 1),
                ('EDGE_WEIGHT_TYPE', 'EUC_2D'),
                ('CAPACITY', int(instance.capacity))):
        lines.append(f'{key} : {value}')

    lines.append('NODE_COORD_SECTION')
    for i, (x, y) in enumerate([instance.depot] + instance.customers):
        x, y = int(x * grid_dim + 0.5), int(y * grid_dim + 0.5)
        lines.append(f'{i + 1}\t{x}\t{y}')

    lines.append('\nDEMAND_SECTION')
    lines.append('\n'.join(f'{i + 1}\t{d}'
                           for i, d in enumerate([0] + instance.demands)))
    lines += ['DEPOT_SECTION', '1', '-1', 'EOF']

    return '\n'.join(lines)


def write_vrp(instance: VRPInstance, filepath: str, grid_dim=GRID_DIM):
    with open(filepath, 'w') as f:
        f.write("\n".join([
            f"{key} : {value}"
            for key, value in (
                ("NAME", os.path.splitext(filepath)[0].split('/')[-1]),
                ("TYPE", "CVRP"),
                ("DIMENSION", instance.n_customers + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", int(instance.capacity))
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

def write_solution(solution: VRPSolution, filename: str):
    with open(filename, 'w') as f:
        f.write(f"DIMENSION {GRID_DIM}\n")
        f.write("TOUR_SECTION\n")
        last = None
        for tour in solution.routes:
            for node in tour[:-1]:
                f.write(f"{node + 1}\n")
        f.write("-1\n")