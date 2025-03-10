'''
Glossary:
    - Table: A DataFrame stored on disk, that can span multiple files, and can be larger-than-ram.
    - Static Table: An inmutable table, once generated will not change.
    - Dynamic Table: A table to which new rows could be added at any point.
    - Transform: A table, that is the result of a data transformation using a table as source.

'''

from .schema import Column as Column, Schema as Schema

from .table import Table as Table

from .transform import Transform as Transform
