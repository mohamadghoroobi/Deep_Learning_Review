"""
Modern GPT - Complete Implementation from Scratch

Features:
- RoPE (Rotary Positional Embedding)
- GQA (Grouped Query Attention)
- SwiGLU Activation
- Pre-Norm Architecture
- RMSNorm
- KV-cache for generation
"""

from .model import ModernGPT, ModernDecoderBlock
from .config import ModelConfig

__all__ = [
    'ModernGPT',
    'ModernDecoderBlock',
    'ModelConfig'
]