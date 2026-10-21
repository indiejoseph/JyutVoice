"""The training helpers are re-exported lazily (PEP 562): they pull in lightning, hydra, rich and matplotlib, which inference (`jyutvoice.utils.onnx_synth`,
`jyutvoice.utils.onnx_infer`, the API image) does not install."""
import importlib

_EXPORTS = {
    "instantiate_callbacks": "jyutvoice.utils.instantiators",
    "instantiate_loggers": "jyutvoice.utils.instantiators",
    "log_hyperparameters": "jyutvoice.utils.logging_utils",
    "get_pylogger": "jyutvoice.utils.pylogger",
    "enforce_tags": "jyutvoice.utils.rich_utils",
    "print_config_tree": "jyutvoice.utils.rich_utils",
    "extras": "jyutvoice.utils.utils",
    "get_metric_value": "jyutvoice.utils.utils",
    "task_wrapper": "jyutvoice.utils.utils",
}


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
