"""
SIGIL -- Sliced Isotropy-Guided Invertible Linearisation
=========================================================
One geometric quantity, learned once per layer, pays for three things:

  1. compression   distribution-matched orthogonal transforms that make weights
                   and KV cache quantisable at lower bit-width  (sigil.rotation)
  2. allocation    the same statistic predicts per-layer distortion, giving a
                   principled bit budget instead of a sensitivity sweep
                   (sigil.allocate)
  3. routing       in an isotropised space, OOD detection collapses to a
                   chi-square test on the squared norm -- one dot product, no
                   extra head, no second forward pass  (sigil.allocate.IsotropyRouter)

Every learned transform is orthogonal and folds exactly into existing weight
matrices, so the deployed graph gains zero operators (sigil.npu).
"""
__version__ = "0.1.0"

from .gof import EppsPulley, UniformCvM, Kurtosis, global_scale          # noqa: F401
from .rotation import (CayleyRotation, ButterflyRotation,                # noqa: F401
                       random_hadamard, fit_rotation)
from .quant import (IntRTN, NFCodebook, quantize_kv, rel_mse,            # noqa: F401
                    inner_product_error, effective_bits)
from .allocate import (allocate_bits, predict_shape_factor,              # noqa: F401
                       validate_surrogate, IsotropyRouter, CascadeConfig)
from .npu import (fold_qk, fold_vo, rope_safe_mask, check_hexagon,       # noqa: F401
                  aihub_export_snippet, geniex_snippet)

__all__ = [n for n in dir() if not n.startswith("_")]
