"""
    This module contains some helpful methods and classes.
"""

from main.nenv.utils.ProcessManager import ProcessManager
from main.nenv.utils.SessionOps import AGENT_OPERATIONS, session_operation
from main.nenv.utils.KillableThread import KillableThread
from main.nenv.utils.ExcelLog import ExcelLog, LogRow
from main.nenv.utils.Move import get_move, get_move_distribution, calculate_move_correlation, calculate_awareness, calculate_behavior_sensitivity
from main.nenv.utils.OSUtils import open_folder
