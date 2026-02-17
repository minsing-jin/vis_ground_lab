"""Model wrappers and registries."""

from vis_ground_lab.models.factory import create_model_wrapper
from vis_ground_lab.models.florence2 import Florence2Wrapper

__all__ = ["Florence2Wrapper", "create_model_wrapper"]
