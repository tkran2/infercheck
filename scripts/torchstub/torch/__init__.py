# Minimal stand-in so convert_hf_to_gguf.py can run --vocab-only without PyTorch (no tensors are read).
class _Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Dummy()
    def __getattr__(self, name): return _Dummy()
    def __mro_entries__(self, bases): return (object,)
def __getattr__(name):
    return _Dummy()
__version__ = "0.0.0-stub"
class inference_mode:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
