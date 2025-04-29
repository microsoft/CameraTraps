"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .det_solver import DetSolver
from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
}