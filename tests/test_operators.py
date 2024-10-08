import copy
from operator import attrgetter
from functools import partial

import pytest

import torch
import torch.nn as nn
from helpers import (set_default_rng, empty_solutions, skipif_module,   # NOQA
                     MissingPlaceholder, complete_solutions)
from context import nlns

from nlns.operators import LNSOperator, ChainedLNSOperator
from nlns.operators.repair import GreedyRepair
from nlns.operators.repair.rl_agent import RLAgentRepair
from nlns.models import VRPActorModel, ResidualGatedGCNModel
try:
    from nlns.operators.repair.scip import SCIPRepair
except ModuleNotFoundError:
    SCIPRepair = MissingPlaceholder('SCIPRepair')

from nlns.operators.destroy import PointDestroy, RandomDestroy, TourDestroy
from nlns.operators.destroy.heatmap import HeatmapDestroy


vrp_actor_model = VRPActorModel()
resgatedgcn_model = nn.DataParallel(ResidualGatedGCNModel())
resgatedgcn_model.load_state_dict(
    torch.load('resgatedgcn100.zip', map_location='cpu')['model_state_dict'])
resgatedgcn_model.eval()


def rlagent_repair_factory():
    return RLAgentRepair(vrp_actor_model)


def heatmap_destroy_factory(percentage):
    return HeatmapDestroy(percentage, resgatedgcn_model)


def chain_destroy_factory(percentage):
    return ChainedLNSOperator([PointDestroy(percentage / 2),
                               RandomDestroy(percentage / 2)])


rlagent_repair_factory.__name__ = 'RLAgentRepair'
heatmap_destroy_factory.__name__ = 'HeatmapDestroy'
chain_destroy_factory.__name__ = 'ChainedDestroy(point+random)'

repair_operators = [
            (GreedyRepair, 42),
            pytest.param(SCIPRepair, 42, marks=[skipif_module('pyscipopt')]),
            (rlagent_repair_factory, 42)
        ]

# Necessary as long as we have the reproducibility bug on SCIPRepair
repair_operators_reproducibility = [
            (GreedyRepair, 42),
            pytest.param(SCIPRepair, 42, marks=[skipif_module('pyscipopt'),
                                                pytest.mark.xfail]),
            (rlagent_repair_factory, 42)
        ]

destroy_operators = [
            (PointDestroy, 42),
            (RandomDestroy, 42),
            (TourDestroy, 42),
            (heatmap_destroy_factory, 42),
            (chain_destroy_factory, 42)
        ]


class IdentityOperator(LNSOperator):
    """Return solutions unchanged. Testing purposes."""

    def __call__(self, solutions):
        return solutions


@pytest.fixture
def identity_operator():
    return IdentityOperator()


class TestLNSOperator:

    def test_init(self, identity_operator, set_default_rng):    # NOQA
        assert identity_operator.rng is nlns.default_rng

    def test_set_random_state(self, identity_operator,
                              set_default_rng, seed=1):         # NOQA
        identity_operator.set_random_state(seed)
        assert identity_operator.rng.getstate() != nlns.default_rng.getstate()


def param_repair(params=repair_operators):
    return pytest.mark.parametrize('operator_type, seed',
                                   params)


def param_destroy(params=destroy_operators):
    return lambda cls: pytest.mark.parametrize('operator_type, seed', params)(
        pytest.mark.parametrize('percentage', [0.2, 0.5, 0.7])(cls))


class TestRepair:

    @param_repair()
    def test_repair(self, operator_type, seed, empty_solutions):        # NOQA
        operator = operator_type()

        operator.set_random_state(seed)
        solutions = operator(empty_solutions)

        assert all(not solution.missing_customers() for solution in solutions)

    @param_repair(repair_operators_reproducibility)
    def test_reproducibility(self, operator_type, seed,
                             empty_solutions):                  # NOQA
        operator = operator_type()

        empty_copy = copy.deepcopy(empty_solutions)

        operator.set_random_state(seed)
        solutions = operator(empty_solutions)

        operator.set_random_state(seed)
        solutions_copy = operator(empty_copy)

        costs = list(map(attrgetter('cost'), solutions))
        costs_copy = list(map(attrgetter('cost'), solutions_copy))

        assert costs == costs_copy


class TestDestroy:

    @param_destroy()
    def test_destroy(self, operator_type, seed, percentage,
                     complete_solutions):                   # NOQA
        operator = operator_type(percentage)

        operator.set_random_state(seed)
        solutions = operator(complete_solutions)

        assert all(solution.missing_customers() for solution in solutions)

    @param_destroy()
    def test_reproducibility(self, operator_type, seed, percentage,
                             complete_solutions,):           # NOQA
        operator = operator_type(percentage)

        complete_copy = copy.deepcopy(complete_solutions)

        operator.set_random_state(seed)
        solutions = operator(complete_solutions)

        operator.set_random_state(seed)
        solutions_copy = operator(complete_copy)

        costs = list(map(attrgetter('cost'), solutions))
        costs_copy = list(map(attrgetter('cost'), solutions_copy))

        assert costs == costs_copy
