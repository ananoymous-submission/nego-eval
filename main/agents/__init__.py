"""Build-in agents in Negotiation ENVironment.

Import each agent directly from its submodule, e.g.:
    from agents.boulware.Boulware import BoulwareAgent

This package's __init__ stays empty intentionally so importing one agent
does not transitively load every other agent (some carry heavy dependencies
like numba).
"""
