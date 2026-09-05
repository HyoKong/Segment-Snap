"""Vendored USDNet evaluator + our wrapper.

`evaluate_semantic_instance.py` is kept BYTE-IDENTICAL to
https://raw.githubusercontent.com/insait-institute/USDNet/master/benchmark/evaluate_semantic_instance.py
so that any divergence from the organisers' scoring is visible in a diff. Same for util.py /
util_3d.py.

That file does `import torch` at line 35 and then never references it (verified: the string
"torch" appears exactly once in the file, and not at all in util*.py). Installing a ~2.5 GB
CUDA torch just to satisfy a dead import is a heavy dependency for nothing, so we register a
placeholder module instead when torch is genuinely absent. If torch IS installed, the real one
is used and this does nothing.
"""
import os as _os
import sys as _sys
import types as _types

# --- numpy 2.x compatibility -------------------------------------------------------------
# The vendored evaluator predates numpy 2 and calls np.in1d, removed in 2.0
# (evaluate_semantic_instance.py:624, on the articulate3d code path — it builds the void mask
# that decides whether an unmatched prediction is charged as a false positive, so it is NOT
# optional). np.isin is the documented drop-in for 1-D inputs, which these are.
#
# np.int / np.float also appear at lines 1284-1286 and 1500-1501 but only inside
# `if dataset == "s3dis"` branches, which articulate3d never enters — left alone deliberately
# so the vendored file stays byte-identical to upstream.
import numpy as _np

if not hasattr(_np, "in1d"):
    _np.in1d = lambda ar1, ar2, **kw: _np.isin(ar1, ar2, **kw).ravel()

# The vendored evaluator does `import benchmark.util as util` / `import benchmark.util_3d`,
# i.e. it expects to live inside USDNet's `benchmark/` package. Rather than restructure our
# tree (or patch the file and lose byte-identity with upstream), publish a synthetic `benchmark`
# package whose search path is this directory, where util.py and util_3d.py already sit.
if "benchmark" not in _sys.modules:
    _pkg = _types.ModuleType("benchmark")
    _pkg.__path__ = [_os.path.dirname(_os.path.abspath(__file__))]
    _pkg.__doc__ = "Synthetic package alias so the vendored USDNet evaluator imports unmodified."
    _sys.modules["benchmark"] = _pkg

try:  # pragma: no cover - trivial
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _stub = _types.ModuleType("torch")
    _stub.__doc__ = (
        "Placeholder injected by arti3d.eval — the vendored evaluator imports torch but never "
        "uses it. Any attribute access is a bug: it means the evaluator started using torch and "
        "the vendored copy must be re-checked against upstream."
    )

    # SciPy's array-API dispatch probes `torch.Tensor` via issubclass on any imported module
    # named "torch" (scipy/_external/array_api_compat/common/_helpers.py:is_torch_array), so the
    # placeholder must answer that probe with a real class that nothing will ever be an instance
    # of. Everything else still raises, so genuine use by the evaluator is caught.
    class _NotATensor:
        """Sentinel: no object is an instance of this."""

    _stub.Tensor = _NotATensor
    _stub.__version__ = "0.0.0+arti3d-placeholder"
    _stub.bool = bool

    def _boom(name):
        raise AttributeError(
            f"arti3d.eval torch placeholder: attribute {name!r} was requested. The vendored "
            "evaluator was not supposed to use torch — re-check it against upstream and install "
            "the real torch."
        )

    _stub.__getattr__ = _boom
    _sys.modules["torch"] = _stub
