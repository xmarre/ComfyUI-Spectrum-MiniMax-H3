from pathlib import Path

path = Path("comfyui_spectrum_h3/core_bsa_compat.py")
text = path.read_text(encoding="utf-8")
replacements = {
    '''    provider_identity: str\n    exact_k_block_range: tuple[int, int]\n''': '''    provider_identity: str\n    q_rows: int\n    kv_rows: int\n    exact_k_block_range: tuple[int, int]\n''',
    '''        provider_identity=CORE_BSA_MEASURE_PROVIDER,\n        exact_k_block_range=tuple(exact_range),\n''': '''        provider_identity=CORE_BSA_MEASURE_PROVIDER,\n        q_rows=seq_len,\n        kv_rows=seq_len,\n        exact_k_block_range=tuple(exact_range),\n''',
    '''        ("rows", len(measure.normalized_request),),\n''': '''        ("rows", measure.q_rows, measure.kv_rows),\n''',
    '''        measure.pool_keys[index][1],\n        measure.pool_keys[index][1],\n        measure.exact_range_digest,\n''': '''        measure.q_rows,\n        measure.kv_rows,\n        measure.exact_range_digest,\n''',
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"generated core_bsa_compat.py: expected one row-identity anchor, found {count}: {old!r}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
