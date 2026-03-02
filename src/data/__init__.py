"""Dataset loaders: HotpotQA, MuSiQue, 2WikiMultiHop."""

from .hotpotqa import HotpotQALoader, load_hotpotqa
from .musique import MuSiQueLoader, load_musique
from .wiki2hop import Wiki2HopLoader, load_2wiki

__all__ = [
    "HotpotQALoader",
    "load_hotpotqa",
    "MuSiQueLoader",
    "load_musique",
    "Wiki2HopLoader",
    "load_2wiki",
]
