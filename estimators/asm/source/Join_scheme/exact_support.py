import re

import numpy as np
import pandas as pd

from logical_tree import parse_logic_tree


def _predicate_mask(tree, table):
    """Evaluate a parsed SQL predicate tree on an AR table's raw arrays."""
    if tree.children:
        children = [_predicate_mask(child, table) for child in tree.children]
        op = tree.op.upper()
        if op == 'AND':
            return np.logical_and.reduce(children)
        if op == 'OR':
            return np.logical_or.reduce(children)
        raise ValueError(f'Unsupported logical operator: {tree.op}')

    col_name = tree.col.split('.', 1)[-1]
    values = np.asarray(table[col_name].data)
    op = tree.op.lower()
    value = tree.value
    not_null = ~pd.isnull(values)

    if op == '=':
        return not_null & (values == value)
    if op == '!=':
        return not_null & (values != value)
    if op == '>':
        return not_null & (values > value)
    if op == '>=':
        return not_null & (values >= value)
    if op == '<':
        return not_null & (values < value)
    if op == '<=':
        return not_null & (values <= value)
    if op == 'is null':
        return ~not_null
    if op == 'is not null':
        return not_null
    if op in ('in', 'not in'):
        candidates = (value if isinstance(value, (list, tuple, np.ndarray))
                      else [value])
        converted = []
        for candidate in candidates:
            try:
                converted.append(float(candidate) if '.' in str(candidate)
                                 else int(candidate))
            except (TypeError, ValueError):
                converted.append(candidate)
        mask = not_null & np.isin(values, converted)
        return ~mask & not_null if op == 'not in' else mask
    if op in ('like', 'not like'):
        pattern = ''.join(
            '.*' if char == '%' else '.' if char == '_' else re.escape(char)
            for char in str(value)
        )
        mask = np.zeros(len(values), dtype=np.bool_)
        mask[not_null] = np.fromiter(
            (re.fullmatch(pattern, str(v)) is not None
             for v in values[not_null]),
            dtype=np.bool_, count=int(not_null.sum()))
        return ~mask & not_null if op == 'not like' else mask
    raise ValueError(f'Unsupported predicate operator: {tree.op}')


def _signature_counts(row_signatures, sample_signatures):
    """Return exact row multiplicities for sampled join-key signatures."""
    if len(row_signatures) == 0:
        return np.zeros(len(sample_signatures), dtype=np.int64)

    rows = np.ascontiguousarray(row_signatures, dtype=np.int32)
    samples = np.ascontiguousarray(sample_signatures, dtype=np.int32)
    assert rows.ndim == samples.ndim == 2
    assert rows.shape[1] == samples.shape[1]

    if rows.shape[1] == 1:
        row_values = rows[:, 0]
        sample_values = samples[:, 0]
        max_value = int(max(row_values.max(), sample_values.max()))
        frequencies = np.bincount(row_values, minlength=max_value + 1)
        return frequencies[sample_values]

    key_dtype = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    row_keys = rows.view(key_dtype).reshape(-1)
    sample_keys = samples.view(key_dtype).reshape(-1)
    unique_keys, counts = np.unique(row_keys, return_counts=True)
    positions = np.searchsorted(unique_keys, sample_keys)
    matched = positions < len(unique_keys)
    matched[matched] &= unique_keys[positions[matched]] == sample_keys[matched]
    result = np.zeros(len(samples), dtype=np.int64)
    result[matched] = counts[positions[matched]]
    return result


def compute_exact_failure_masks(factors, tables_all, ncs, query_predicate,
                                schema, sample_size):
    """Return exact per-alias failures for ASM's sampled join-key buckets.

    The check uses the factorized tuple representation consumed by the AR.
    Consequently, when ASM samples only factor 0 of a factorized join key, the
    support event is evaluated at exactly that bucket granularity.
    """
    failure_masks = dict()
    for alias, factor in factors.items():
        table_name = tables_all[alias]
        nc = ncs[table_name]
        predicate = query_predicate[alias][1]

        if predicate:
            tree = parse_logic_tree(predicate, alias, table_name, schema)
            predicate_mask = _predicate_mask(tree, nc.table)
        else:
            predicate_mask = np.ones(nc.table.cardinality, dtype=np.bool_)

        train_data = nc.train_data
        tuples_np = (train_data.factorized_tuples_np
                     if hasattr(train_data, 'factorized_tuples_np')
                     else train_data.tuples_np)
        assert len(predicate_mask) == len(tuples_np)

        sampled_indexes = sorted(factor.join_sample_bins)
        if sampled_indexes:
            row_signatures = tuples_np[predicate_mask][:, sampled_indexes]
            sample_signatures = np.column_stack(
                [factor.join_sample_bins[idx] for idx in sampled_indexes]
            )
            failure_masks[alias] = (
                _signature_counts(row_signatures, sample_signatures) == 0
            )
        else:
            # No sampled join key means only an empty local predicate can fail.
            failure_masks[alias] = np.full(
                sample_size, not predicate_mask.any(), dtype=np.bool_)

    return failure_masks
