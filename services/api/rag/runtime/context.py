from dataclasses import dataclass
from contextvars import ContextVar
from core.models import ProviderConfig, KeysConfig

@dataclass
class Runtime:
    provider: ProviderConfig
    keys: KeysConfig

_runtime: ContextVar[Runtime] = ContextVar("runtime")

def set_runtime(provider: ProviderConfig, keys: KeysConfig) -> None:
    _runtime.set(Runtime(provider=provider, keys=keys))

def get_runtime() -> Runtime:
    return _runtime.get()

# convenience
class _RuntimeProxy:
    @property
    def provider(self):
        return get_runtime().provider
    @property
    def keys(self):
        return get_runtime().keys

runtime = _RuntimeProxy()
