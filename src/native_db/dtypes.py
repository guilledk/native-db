'''
# Overview

The key to expanding `polars` capabilities for our larger than memory datasets
and iterative (streaming) algorithms is being able to provide extra information
about or table schemas apart from just column names & types.

Expanding schema information can also allow for automatic optimization of
arrow ipc & parquet settings like row group sizes, usage of dictionaries etc.

# Extended `polars.DataType`

## Keyword

By adding a new short string type `Keyword` (name borrowed from elasticsearch),
we can automatically infer that using a dictionary on that column on parquets
is a good idea, as well as that we want to optimize querying by that column.

## Mono32 & Mono64

A common usecase of numeric data types is a monotonic identifier, that allows
us to infer ordering and partitioning semantics.

# TypeHints

In order to simplify finetuned control over table settings, schemas support an
optional `TypeHints` parameter, where users can convey additional information
about the column outside the regular type system:

    - Optionality
    - Uniqueness
    - Sort order
    - Dictionary use override
    - Expected avg size for dynamic size columns (allows row group settings
      inference).

'''

from __future__ import annotations

from inspect import isclass
from types import UnionType
from typing import Any, Literal, TypeGuard

import msgspec
import polars as pl
from polars.datatypes.classes import NumericType, classinstmethod
from polars._typing import PythonDataType, PolarsDataType

from native_db.structs import FrozenStruct


class Keyword(pl.DataType):
    '''
    For string columns that are short and we might want to search by.

    '''

    @classinstmethod
    def to_python(cls) -> PythonDataType:
        return str

    @classinstmethod
    def fallback_type(cls) -> type[pl.DataType]:
        return pl.String


IntegerSizes = Literal[4, 8, 16]


class Mono(NumericType):
    size: IntegerSizes

    def __init__(self, size: IntegerSizes) -> None:
        self.size = size

    @classinstmethod
    def to_python(cls) -> PythonDataType:
        return int

    def __eq__(self, other: PolarsDataType) -> bool:
        return isinstance(other, Mono) and other.size == self.size

    def __hash__(self) -> int:
        return hash((self.__class__, self.size))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f'{class_name}(size={self.size})'

    @classinstmethod
    def fallback_type(cls) -> type[pl.DataType]:
        match cls.size:
            case 4:
                return pl.UInt32

            case 8:
                return pl.UInt64

            case 16:
                return pl.Int128



# type hints

# custom sort literals we accept
SortTypes = Literal['asc', 'desc']

# simple type kind distinction, can be used to infer settings
TypeKind = Literal['numeric', 'string', 'nested']

# our custom extended types
DataTypeCustom = type[Keyword] | Mono

# all types, polars std + our custom types
DataTypeExt = type[pl.DataType] | pl.DataType | DataTypeCustom


# type sets

# custom types that are fixed size
custom_types_fixed_size: tuple[type[pl.DataType], ...] = (Mono, )

# all custom types
custom_types: tuple[type[pl.DataType], ...] = (
    Keyword,
    *custom_types_fixed_size,
)

# all string like types, polars + custom
all_string_likes: tuple[type[pl.DataType], ...] = (
    pl.Binary,
    pl.String,
    pl.Utf8,
    Keyword,
)


# byte sizes for every polars / arrow type
arrow_type_fixed_sizes: dict[type[pl.DataType], int] = {
    pl.UInt8: 1,
    pl.UInt16: 2,
    pl.UInt32: 4,
    pl.UInt64: 8,
    pl.Int8: 1,
    pl.Int16: 2,
    pl.Int32: 4,
    pl.Int64: 8,
    pl.Int128: 16,
    pl.Float32: 4,
    pl.Float64: 8,
    pl.Decimal: 16,
    pl.Boolean: 1,
    pl.Date: 4,
    pl.Time: 8,
    pl.Datetime: 8,
    pl.Duration: 8,
}


# all fixed size types, polars + custom
all_fixed_size_types: tuple[type[pl.DataType], ...] = (
    tuple(arrow_type_fixed_sizes.keys()) + custom_types_fixed_size
)


def class_of(dtype: DataTypeExt) -> type[pl.DataType]:
    return type(dtype) if not isclass(dtype) else dtype


# type predicates


def is_fixed_size(dtype: DataTypeExt) -> bool:
    return class_of(dtype) in all_fixed_size_types


def is_custom_type(dtype: DataTypeExt) -> TypeGuard[DataTypeCustom]:
    return class_of(dtype) in custom_types


# type size estimation


def maybe_fixed_type_size(dtype: DataTypeExt) -> int | None:
    '''
    Given any non-nested polars type, maybe obtain its fixed size if
    applicable.

    '''
    if is_custom_type(dtype):
        dtype = dtype.fallback_type()

    # search size or return none
    return arrow_type_fixed_sizes.get(class_of(dtype), None)


def avg_type_size(dtype: DataTypeExt, hints: TypeHints) -> int:
    '''
    Given a polars data type, estimate its byte size, using the hint system
    for types that have dynamic size component.

    '''
    if not dtype.is_nested():
        # fixed size type?
        fixed_size = maybe_fixed_type_size(dtype)
        if fixed_size:
            return fixed_size

        # dynamic string like?
        if dtype in all_string_likes:
            return 8 + hints.avg_str_size

    else:
        # maybe use nested hints
        nested_hints = hints if not hints.nested else hints.nested

        if isinstance(dtype, pl.Struct):
            # recursive per struct field size calculation and sumation
            inner_size = sum(
                (
                    avg_type_size(field.dtype, nested_hints)
                    for field in dtype.fields
                )
            )
            return 8 + inner_size

        if isinstance(dtype, pl.List | pl.Array):
            # recursive inner type size calculation times avg list size hint
            return 8 + (
                avg_type_size(dtype.inner, nested_hints) * hints.avg_list_size
            )

    raise TypeError(f'Unknown avg size for type {dtype}')


# additional type metadata


def type_kind(dtype: DataTypeExt) -> TypeKind:
    '''
    Given any data type we support on columns (standard and custom) obtain its
    "kind" which is usefull for inference of some settings.

    '''
    if dtype.is_nested():
        return 'nested'

    if is_fixed_size(dtype):
        # all fixed size types are numeric
        return 'numeric'

    if dtype in all_string_likes:
        # string likes
        return 'string'

    raise TypeError(f'Unknown kind for type {dtype}')


def should_use_dictionary(dtype: DataTypeExt) -> bool:
    '''
    Given any data type we support on columns (standard and custom) figure out
    if it might benefit from using dictionaries in parquet row group metadata.

    '''
    return dtype in (pl.String, pl.Utf8, Keyword)


class TypeHints(msgspec.Struct, frozen=True):
    # write time hints
    optional: bool = False
    unique: bool = False
    sort: SortTypes | None = None
    use_dictionary: bool | None = None

    # for size avg calculation
    avg_str_size: int = 0
    avg_list_size: int = 0

    nested: TypeHints | None = None


def py_type_for(dtype: DataTypeExt, *, hints: TypeHints = TypeHints()) -> type:
    '''
    Given any data type we support on columns (standard and custom) obtain
    which python type it would map to.

    Useful for example when generating a `msgspec.Struct` from a `Table`
    schema.

    '''
    py_type: PythonDataType | UnionType = dtype.to_python()

    if not dtype.is_nested():
        py_type = dtype.to_python()

    else:
        nested_hints = hints if not hints.nested else hints.nested
        if isinstance(dtype, pl.Struct):
            py_type = dict[str, any]

        if isinstance(dtype, pl.List | pl.Array):
            list_type: type[list] = list[
                py_type_for(dtype.inner, hints=nested_hints)
            ]
            py_type = list_type

    if hints.optional:
        py_type = py_type | None

    return py_type


# data type serialization

# a tiny, serializable dtype tag
DTypeTag = Literal[
    'u8',
    'u16',
    'u32',
    'u64',
    'i8',
    'i16',
    'i32',
    'i64',
    'i128',
    'f32',
    'f64',
    'decimal',
    'bool',
    'date',
    'time',
    'datetime',
    'duration',
    'binary',
    'string',  # polars string
    'keyword',  # custom Keyword
    'mono', # custom monotonic

    # nested sequences
    'list',
    'array'
]

# Maps between runtime dtype classes and tags
dtype_tag_map: dict[DataTypeExt, DTypeTag] = {
    pl.UInt8: 'u8',
    pl.UInt16: 'u16',
    pl.UInt32: 'u32',
    pl.UInt64: 'u64',
    pl.Int8: 'i8',
    pl.Int16: 'i16',
    pl.Int32: 'i32',
    pl.Int64: 'i64',
    pl.Int128: 'i128',
    pl.Float32: 'f32',
    pl.Float64: 'f64',
    pl.Decimal: 'decimal',
    pl.Boolean: 'bool',
    pl.Date: 'date',
    pl.Time: 'time',
    pl.Datetime: 'datetime',
    pl.Duration: 'duration',
    pl.Binary: 'binary',
    pl.String: 'string',
    Keyword: 'keyword',
    Mono: 'mono',
    pl.List: 'list',
    pl.Array: 'array'
}

# inverse of dtype_tag_map
tag_dtype_map: dict[DTypeTag, DataTypeExt] = {
    v: k for (k, v) in dtype_tag_map.items()
}


class DataTypeMeta(FrozenStruct, frozen=True):
    tag: DTypeTag
    kwargs: dict[str, Any] = {}

    @staticmethod
    def from_dtype(dtype: DataTypeExt) -> DataTypeMeta:
        kwargs = {}
        match dtype:
            case Mono():
                kwargs['size'] = dtype.size

            case pl.Decimal():
                kwargs['precision'] = dtype.precision
                kwargs['scale'] = dtype.scale

            case pl.Datetime():
                kwargs['time_unit'] = dtype.time_unit
                kwargs['time_zone'] = dtype.time_zone

            case pl.Duration():
                kwargs['time_unit'] = dtype.time_unit

            case pl.List() | pl.Array():
                kwargs['inner'] = DataTypeMeta.from_dtype(dtype.inner)

            case _ if dtype.is_nested():
                raise NotImplementedError(f'Only List or Array nested types supported, got: {dtype}')

        return DataTypeMeta(tag=dtype_tag_map[class_of(dtype)], kwargs=kwargs)

    def decode(self) -> DataTypeExt:
        cls = tag_dtype_map[self.tag]

        match cls:
            case pl.List | pl.Array:
                inner_meta = DataTypeMeta.convert(self.kwargs['inner'])
                cls = cls(inner_meta.decode())

            case _ if cls in (
                Mono,
                pl.Decimal,
                pl.Datetime,
                pl.Duration
            ):
                cls = cls(**self.kwargs)

        return cls
