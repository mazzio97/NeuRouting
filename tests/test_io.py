import pytest

from helpers import get_filename
from nlns.utils.vrp_io import read_vrp_str, read_vrp, GRID_DIM


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
