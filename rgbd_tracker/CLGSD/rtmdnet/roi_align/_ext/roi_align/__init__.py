import torch
# from torch.utils.ffi import _wrap_function
from ._roi_align import lib as _lib, ffi as _ffi
from functools import wraps


_torch_to_cffi = {}
_cffi_to_torch = {}

def _wrap_function(function, ffi):
    @wraps(function)
    def safe_call(*args, **kwargs):
        args = tuple(ffi.cast(_torch_to_cffi.get(arg.type(), 'void') + '*', arg._cdata)
                     if isinstance(arg, torch.Tensor) or torch.is_storage(arg)
                     else arg
                     for arg in args)
        args = (function,) + args
        result = torch._C._safe_call(*args, **kwargs)
        if isinstance(result, ffi.CData):
            typeof = ffi.typeof(result)
            if typeof.kind == 'pointer':
                cdata = int(ffi.cast('uintptr_t', result))
                cname = typeof.item.cname
                if cname in _cffi_to_torch:
                    # TODO: Maybe there is a less janky way to eval
                    # off of this
                    return eval(_cffi_to_torch[cname])(cdata=cdata)
        return result
    return safe_call


__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())
