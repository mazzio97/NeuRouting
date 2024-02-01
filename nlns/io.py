import os
from typing import List, Optional

import numpy as np
from more_itertools import split_after

from nlns.instances import VRPInstance, VRPSolution, Route


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
        depot=(_norm(locations[0][1], grid_dim),
               _norm(locations[0][2], grid_dim)),
        customers=[(_norm(loc[1], grid_dim),
                    _norm(loc[2], grid_dim)) for loc in locations[1:]],
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


def write_vrp(instance: VRPInstance, filepath: Optional[str] = None, file=None,
              name: Optional[str] = None, grid_dim=GRID_DIM):
    """Write a VRP instance to string.

    Args:
        instance: The input instance.
        filepath: Filename or path to save the instance to.
        file: A filelike object to write to. If given, overrides
            ``filepath``
        name: A name for the instance. If omitted, the filepath is used
            to determine it (base name of the file is used). If ``file``
            is used instead of ``filepath``, it is mandatory to specify
            the name explicitly.
        grid_dim: See :func:`vrp_read_str`.
    """
    # Handle instance name
    if filepath is None and name is None:
        raise ValueError('If no filepath is specified, a name is mandatory.')

    if filepath is not None:
        instance_name = os.path.splitext(os.path.basename(filepath))[0]

    if name is not None:
        instance_name = name

    # Generate and dump string
    vrp_string = write_vrp_str(instance, instance_name, grid_dim=grid_dim)

    if file is None:
        with open(filepath, 'w') as fout:
            fout.write(vrp_string)
    else:
        file.write(vrp_string)


def read_routes_str(vrp_solution_str: str, num_nodes: int) -> List[List[int]]:
    """Read VRP solution routes from string.

    Args:
        vrp_solution_str: Input string.
        num_nodes: Expected number of customers.
    Return:
        A list of lists representing the routes.
    """
    tour = []
    # dimension = 0
    started = False
    for line in vrp_solution_str.splitlines():
        if started:
            loc = int(line)
            if loc == -1:
                break
            tour.append(loc)
        # if line.startswith('DIMENSION'):
        #     dimension = int(line.split()[-1])

        if line.startswith('TOUR_SECTION'):
            started = True

    # Subtract 1 as depot is 1 and should be 0
    tour = np.array(tour).astype(int) - 1
    # Any nodes above the number of nodes there are is also depot
    tour[tour > num_nodes] = 0
    tour = tour[1:].tolist() + [0]
    return list([0] + t for t in split_after(tour, lambda x: x == 0))


def read_routes(num_nodes: int, filepath: str = '',
                file=None) -> List[List[int]]:
    """Read VRP routes from file.

    Args:
        num_nodes: Expected number of customers.
        filepath: Filename or path to read the instance from.
        file: A filelike object to read from. If given, overrides
            ``filepath``

    Returns:
        A list of lists representing the routes.
    """
    if file is None:
        with open(filepath) as fin:
            routes_str = fin.read()
    else:
        routes_str = file.read()

    return read_routes_str(routes_str, num_nodes)


def read_solution(instance: VRPInstance, filepath: str = '',
                  file=None) -> VRPSolution:
    """Instantiate a solution given an instance and a solution file.

    Args:
        instance: The instance for which the solution is loaded.
            Solution files contain only routes informations, so that an
            instance must be specified explicitly for them to be
            interpretable.
        filepath: Filename or path to read the instance from.
        file: A filelike object to read from. If given, overrides
            ``filepath``

    Returns:
        A solution populated with routes from the given file,
        referencing ``instance``.
    """
    route_lists = read_routes(instance.n_customers, filepath, file)
    routes = [Route(t, instance) for t in route_lists]
    return VRPSolution(instance, routes)


def write_routes_str(solution: VRPSolution, grid_dim: int = GRID_DIM) -> str:
    """Write VRP solution routes to string.

    Args:
        solution: Input solution.
        num_nodes: Expected number of customers.
    Return:
        A string representing the encoded routes of the solution.
        Instance information is not preserved in this encoding. In case
        saving both an instance and its solution is needed, consider
        dumping the instance on a separate string/file, e.g. with
        :func:`write_vrp_str` or :func:`write_vrp`.
    """
    lines = []
    lines.append(f'DIMENSION {grid_dim}\nTOUR_SECTION')

    for tour in solution.routes:
        for node in tour[:-1]:
            lines.append(f'{node + 1}')
    lines.append('-1')

    return '\n'.join(lines)


def write_routes(solution: VRPSolution, filepath: str = '', file=None,
                 grid_dim: int = GRID_DIM):
    """Write VRP solution routes to file.

    Args:
        solution: Input solution.
        num_nodes: Expected number of customers.
    """
    solution_str = write_routes_str(solution, grid_dim)

    if file is None:
        with open(filepath) as fin:
            fin.write(solution_str)
    else:
        file.write(solution_str)
