"""Crystallographic structure intake and validation.

This package holds everything needed to turn a user-supplied input into a
validated crystallographic model: input resolution, RCSB download, file
reading, unit-cell and space-group accessors, and asymmetric-unit content
validation. It contains no RABDAM science and no E-path analysis, so
`rabdam` stays a faithful reimplementation of the published BDamage
calculation and nothing else.
"""
