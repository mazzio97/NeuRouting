import numpy as np

from instances import VRPSolution
from nlns import RepairProcedure
from nlns.initial.nearest_neighbor import closest_locations


class GreedyRepair(RepairProcedure):
    def __call__(self, solution: VRPSolution):
        instance = solution.instance
        while len(solution.missing_customers()) > 0:
            missing = np.array(solution.missing_customers())

            mask = np.array([False] * (instance.n_customers + 1), dtype=bool)
            mask[missing] = True
            cust = np.random.choice(missing)
            cust_route = solution.get_customer_route(cust)
            self_head = cust_route[0] == cust
            mask[cust_route[0]] = False
            mask[cust_route[-1]] = False
            mask[0] = True
            nearest = closest_locations(instance, cust, mask=mask)

            for neigh in nearest:
                if neigh == 0:
                    cust_route.append_route([neigh], self_head, True)
                    break
                neigh_route = solution.get_customer_route(neigh)
                neigh_head = neigh_route[0] == neigh
                if cust_route.total_demand() + neigh_route.total_demand() <= instance.capacity:
                    cust_route.append_route(neigh_route, self_head, neigh_head)
                    solution.routes.remove(neigh_route)
                    break
        return solution
