from datetime import date, datetime
import json

from pathlib import Path
from typing import Any, Self, Type
from uuid import UUID

import msgspec


class ExtendedEncoder(json.JSONEncoder):
    '''
    stdlib JSONEncoder that supports date types & `pathlib.Path`

    '''
    def default(self, o):
        if isinstance(o, date):
            return o.strftime('%Y-%m-%d')

        if isinstance(o, datetime):
            return o.isoformat()

        if isinstance(o, Path | UUID):
            return str(o)

        return super().default(o)


def ext_enc_hook(obj: Any) -> Any:
    '''
    Extended encoder hook for msgspec to make `Struct.to_dict()` work with
    additional stdlib types.

    '''
    match obj:
        case Path():
            return str(obj)


def ext_dec_hook(type: Type, obj: Any) -> Any:
    """Given a type in a schema, convert ``obj`` (composed of natively
    supported objects) into an object of type ``type``.

    Any `TypeError` or `ValueError` exceptions raised by this method will
    be considered "user facing" and converted into a `ValidationError` with
    additional context. All other exceptions will be raised directly.
    """
    match type:
        case Path:
            return Path(obj)


class _Struct:
    @classmethod
    def from_other(cls, other: Self, **kwargs) -> Self:
        params = other.to_dict()
        params.update(kwargs)
        return cls(**params)

    @classmethod
    def from_json(cls, s: str | bytes) -> Self:
        return msgspec.json.decode(s, type=cls, dec_hook=ext_dec_hook)

    @classmethod
    def from_bytes(cls, raw: bytes) -> Self:
        return msgspec.msgpack.decode(raw, type=cls, dec_hook=ext_dec_hook)

    def encode(self) -> bytes:
        return msgspec.msgpack.encode(self, enc_hook=ext_enc_hook)

    @classmethod
    def convert(cls, obj: Any) -> Self:
        return msgspec.convert(obj, type=cls)

    def to_dict(self) -> dict:
        return msgspec.to_builtins(self, enc_hook=ext_enc_hook)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), cls=ExtendedEncoder, **kwargs)



class Struct(msgspec.Struct, _Struct): ...


class FrozenStruct(msgspec.Struct, _Struct, frozen=True): ...
