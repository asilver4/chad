"""chad — a local MLX-backed, Claude-Code-style coding agent.

A flat collection of cooperating modules behind one console script (``chad``):
the inference engine, the tool layer, the agent loop, and the terminal UI.
"""

__version__ = "1.0.5"

# Tuned MLX runtime vars have to be in the environment before mlx's backend
# initializes, which rules out Engine.load(). chad imports mlx lazily inside
# functions, so package import is the last hook that is still early enough.
from .mlx_env import apply_defaults as _apply_mlx_defaults  # noqa: E402

_apply_mlx_defaults()
