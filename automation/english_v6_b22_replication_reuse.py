"""Reuse unchanged B22 function bodies with an explicit replica module profile.

This does not edit or reload either primary entry point. Imported first-party
code remains part of the complete captured producer. Each inherited function
keeps its exact code object; only its module globals point to the explicit
replica constants and override functions. No model, file or network work runs.
"""
from types import FunctionType


def inherit(original, namespace):
    for name, value in vars(original).items():
        if not name.startswith('__'):
            namespace.setdefault(name, value)
    for name, value in vars(original).items():
        if isinstance(value, FunctionType) and value.__module__ == original.__name__:
            copied = FunctionType(value.__code__, namespace, name, value.__defaults__, value.__closure__)
            copied.__kwdefaults__ = value.__kwdefaults__
            copied.__annotations__ = value.__annotations__
            copied.__module__ = namespace['__name__']
            namespace[name] = copied
