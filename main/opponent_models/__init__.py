"""
    This module contains the Opponent Models
"""

import typing

from . import AbstractOpponentModel

OpponentModelClass = typing.TypeVar('OpponentModelClass', bound=AbstractOpponentModel.__class__)
"""
    Type variable of AbstractOpponentModel class to declare a type for a variable
"""

from .EstimatedPreference import EstimatedPreference
from .ClassicFrequencyOpponentModel import ClassicFrequencyOpponentModel
from .WindowedFrequencyOpponentModel import WindowedFrequencyOpponentModel
from .BayesianOpponentModel import BayesianOpponentModel
from .ConflictBasedOpponentModel import ConflictBasedOpponentModel
