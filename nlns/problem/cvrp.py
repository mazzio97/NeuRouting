from typing import List, Tuple, Dict

from nlns.instances import Coord
from nlns.problem.vrp import VRPInstance

class CVRPInstance(VRPInstance):
    def __init__(self, depot: Coord, customers: List[Coord], demands: List[int], capacity: int):
        """
        Extention to VRP where each vehicle has a maximum allowed capacity.

        Args:
            depot (Coord): X and Y coordinate of the depot. Every vehicle route
                starts and ends at the depot.
            customers (List[Coord]): Coordinates of the customers
            demands (List[int]): Demands of each customer, expressed as an integer.
            capacity (int): Capacity of a vehicle.
        """
        super().__init__(depot, customers, demands)
        self.capacity = capacity
