# Compatibility shim for old checkpoints that expect 'callbacks' module
# This redirects to src.callbacks
from src.callbacks.rllib_callbacks import *
from src.callbacks.papers_metrics_callback import *
try:
    from src.callbacks.debug_actions_callback import *
except ImportError:
    pass
