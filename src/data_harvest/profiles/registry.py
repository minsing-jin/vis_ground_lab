"""Profile registry with @register_profile decorator and auto-discovery."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Type

from data_harvest.profiles.base_profile import GameProfile

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[GameProfile]] = {}


def register_profile(cls: Type[GameProfile]) -> Type[GameProfile]:
    """Class decorator to register a GameProfile subclass."""
    instance = cls()
    name = instance.name
    _REGISTRY[name] = cls
    logger.debug("Registered game profile: %s", name)
    return cls


def get_profile(name: str) -> GameProfile:
    """Instantiate and return a registered profile by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown game profile '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_profiles() -> list[str]:
    """Return all registered profile names."""
    return sorted(_REGISTRY.keys())


def discover_profiles() -> None:
    """Auto-discover profile modules in the profiles package."""
    import data_harvest.profiles as pkg

    for finder, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        if module_name.startswith("_") or module_name in ("base_profile", "registry"):
            continue
        try:
            importlib.import_module(f"data_harvest.profiles.{module_name}")
        except Exception:
            logger.warning("Failed to import profile module: %s", module_name, exc_info=True)
