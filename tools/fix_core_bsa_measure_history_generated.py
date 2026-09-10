from pathlib import Path

path = Path("comfyui_spectrum_h3/core_bsa_compat.py")
text = path.read_text(encoding="utf-8")
replacements = {
    '''    provider_identity: str\n    exact_k_block_range: tuple[int, int]\n''': '''    provider_identity: str\n    q_rows: int\n    kv_rows: int\n    exact_k_block_range: tuple[int, int]\n''',
    '''        provider_identity=CORE_BSA_MEASURE_PROVIDER,\n        exact_k_block_range=tuple(exact_range),\n''': '''        provider_identity=CORE_BSA_MEASURE_PROVIDER,\n        q_rows=seq_len,\n        kv_rows=seq_len,\n        exact_k_block_range=tuple(exact_range),\n''',
    '''        ("rows", len(measure.normalized_request),),\n''': '''        ("rows", measure.q_rows, measure.kv_rows),\n''',
    '''        measure.pool_keys[index][1],\n        measure.pool_keys[index][1],\n        measure.exact_range_digest,\n''': '''        measure.q_rows,\n        measure.kv_rows,\n        measure.exact_range_digest,\n''',
    '''def _pool_key(audit: CoreBSAAudit, index: int) -> tuple[Any, ...]:\n    if audit.measure is not None:\n        return audit.measure.pool_keys[index]\n''': '''def _pool_key(audit: CoreBSAAudit, index: int) -> tuple[Any, ...]:\n    measure = getattr(audit, "measure", None)\n    if measure is not None:\n        return measure.pool_keys[index]\n''',
    '''    measure = audit.measure\n    request = call_options.get(ATTENTION_MEASURE_KEY, _MISSING)\n''': '''    measure = getattr(audit, "measure", None)\n    request = call_options.get(ATTENTION_MEASURE_KEY, _MISSING)\n''',
    '''    if audit.measure is None:\n        return receipt\n    return (\n        *receipt,\n        (\n            ATTENTION_MEASURE_KEY,\n            *_measure_receipt_fields(\n                audit.measure,\n''': '''    measure = getattr(audit, "measure", None)\n    if measure is None:\n        return receipt\n    return (\n        *receipt,\n        (\n            ATTENTION_MEASURE_KEY,\n            *_measure_receipt_fields(\n                measure,\n''',
    '''            expected_route, expected_sink, expected_sink_q = audit.route_specs[index]\n''': '''            expected_route, _expected_sink, _expected_sink_q = audit.route_specs[index]\n''',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"generated core_bsa_compat.py: expected one fix anchor, found {count}: {old!r}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

path = Path("comfyui_spectrum_h3/core_bsa_flow_compat.py")
text = path.read_text(encoding="utf-8")
old = '''            expected_route, expected_sink, expected_sink_q = audit.route_specs[index]\n'''
new = '''            expected_route, _expected_sink, _expected_sink_q = audit.route_specs[index]\n'''
if text.count(old) != 1:
    raise SystemExit("generated core_bsa_flow_compat.py: expected one receipt unpack anchor")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
