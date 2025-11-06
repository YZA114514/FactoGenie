"""Multi-material assembly simulation demo.

This package contains a minimal SimPy-based engine that models
multi-material assembly processes (with assembly stations,
material routes, and sources) and a simple example configuration.
"""

from .engine import (
    AssemblyConfig,
    AssemblySim,
    RouteConfig,
    SourceConfig,
    TransporterConfig,
)

__all__ = [
    "AssemblySim",
    "AssemblyConfig",
    "RouteConfig",
    "SourceConfig",
    "TransporterConfig",
]
