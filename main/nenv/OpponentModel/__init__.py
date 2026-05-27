"""
    This module contains whole components of Opponent Model in Negotiation ENVironment.
"""

import typing

from main.nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel

OpponentModelClass = typing.TypeVar('OpponentModelClass', bound=AbstractOpponentModel.__class__)
"""
    Type variable of AbstractOpponentModel class to declare a type for a variable
"""

from main.nenv.OpponentModel.EstimatedPreference import EstimatedPreference
from main.nenv.OpponentModel.ClassicFrequencyOpponentModel import ClassicFrequencyOpponentModel
from main.nenv.OpponentModel.WindowedFrequencyOpponentModel import WindowedFrequencyOpponentModel
from main.nenv.OpponentModel.BayesianOpponentModel import BayesianOpponentModel
from main.nenv.OpponentModel.ConflictBasedOpponentModel import ConflictBasedOpponentModel
from main.nenv.OpponentModel.CUHKOpponentModel import CUHKOpponentModel
from main.nenv.OpponentModel.StepwiseCOMBOpponentModel import StepwiseCOMBOpponentModel
from main.nenv.OpponentModel.ExpectationCOMBOpponentModel import ExpectationCOMBOpponentModel
from main.nenv.OpponentModel.RegressionCOMBOpponentModel import RegressionCOMBOpponentModel
from main.nenv.OpponentModel.UniformEstimatedPreference import UniformEstimatedPreference
from main.nenv.OpponentModel.CBOMEstimatedPreference import CBOMEstimatedPreference