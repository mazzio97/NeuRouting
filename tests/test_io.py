import pytest
from io import StringIO

from helpers import get_filename
from nlns.utils.vrp_io import (read_vrp_str, read_vrp, write_vrp_str,
                               write_vrp, read_routes_str, read_routes,
                               read_solution,
                               GRID_DIM)


INSTANCE_STRING = """NAME : instance
TYPE : CVRP
DIMENSION : 6
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 20
NODE_COORD_SECTION
1	71485	56286
2	72322	41193
3	78385	16526
4	49710	84061
5	1458	81375
6	16832	40318

DEMAND_SECTION
1	0
2	4
3	9
4	5
5	1
6	3
DEPOT_SECTION
1
-1
EOF
"""

SOLUTION_STRING = """DIMENSION 100000
TOUR_SECTION
1
2
3
6
5
1
4
-1
"""

SOLUTION_ROUTES = [[0, 1, 2, 5, 4, 0], [0, 3, 0]]


@pytest.mark.parametrize('string, capacity, n_customers',
                         [(INSTANCE_STRING, 20, 5)])
def test_read_vrp_str(string, capacity, n_customers):
    instance = read_vrp_str(INSTANCE_STRING, GRID_DIM)

    assert instance.capacity == capacity
    assert len(instance.customers) == n_customers


@pytest.mark.parametrize('filename, capacity, n_customers',
                         [(get_filename('instance.vrp'), 20, 5)])
def test_read_vrp(filename, capacity, n_customers):
    instance = read_vrp(filename, grid_dim=GRID_DIM)

    assert instance.capacity == capacity
    assert len(instance.customers) == n_customers


@pytest.mark.parametrize('string, name, capacity, n_customers',
                         [(INSTANCE_STRING, 'instance', 20, 5)])
def test_write_vrp_str(string, name, capacity, n_customers):
    # For the moment, use the read function to generate the instance
    instance = read_vrp_str(INSTANCE_STRING, GRID_DIM)
    vrp_string = write_vrp_str(instance, name, GRID_DIM)

    assert vrp_string.strip() == string.strip()


@pytest.mark.parametrize('name, string',
                         [('instance', INSTANCE_STRING)])
def test_write_vrp(name, string):
    # For the moment, use the read function to generate the instance
    instance = read_vrp_str(string, grid_dim=GRID_DIM)
    buffer = StringIO()

    write_vrp(instance, file=buffer, name=name)

    buffer.seek(0)
    assert buffer.read().strip() == string.strip()


@pytest.mark.parametrize('string, num_nodes, routes',
                         [(SOLUTION_STRING, 5, SOLUTION_ROUTES)])
def test_read_routes_str(string, num_nodes, routes):
    new_routes = read_routes_str(string, num_nodes)

    for new_route, route in zip(new_routes, routes):
        assert new_route == route


@pytest.mark.parametrize('filename, num_nodes, routes',
                         [(get_filename('solution.sol'), 5, SOLUTION_ROUTES)])
def test_read_routes(filename, num_nodes, routes):
    new_routes = read_routes(num_nodes, filename)

    for new_route, route in zip(new_routes, routes):
        assert new_route == route


@pytest.mark.parametrize('filename, instance_string, routes',
                         [(get_filename('solution.sol'), INSTANCE_STRING,
                           SOLUTION_ROUTES)])
def test_read_solution(filename, instance_string, routes):
    instance = read_vrp_str(instance_string)

    solution = read_solution(instance, filename)

    for new_route, route in zip(solution.routes, routes):
        assert new_route == route
