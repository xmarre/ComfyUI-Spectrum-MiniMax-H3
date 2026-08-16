from pathlib import Path

path = Path("README.md")
text = path.read_text(encoding="utf-8")
anchor = "Spectrum is an **approximate accelerator**. Forecasted steps change the denoising trajectory, so output can differ from native H3 even with the same seed and workflow.\n\n"
section = '''## v0.2.15: H3 Continuum interoperability\n\nv0.2.15 adds first-class interoperability with **H3 Continuum** through its optional API v1 actual-prefix request. Continuation chunks can require their initial solver steps to remain real H3 transformer evaluations; Spectrum treats that as a run-local scheduling constraint without adding a hard dependency on Continuum.\n\nThe prefix applies to ordinary sampling, single-pass fallback, and the offline-smoothing first pass, while transformer-free offline replay remains prefix-free. Invalid, inactive, malformed, negative, or unknown-API metadata leaves normal Spectrum behavior unchanged. Prefix state is scoped to one sampling call and cannot leak into later continuation chunks.\n\nThe interop composes with Diff-Aid's external-patch transition contract. If a Diff-Aid hard transition lands on an already-prefix-protected step, the transition is observed without adding a duplicate H3 evaluation.\n\nThis release also fixes the native ER-SDE edge case exposed by a two-step Continuum prefix. When the first forecast follows consecutive exact actual anchors, Spectrum now uses the newest exact solver-space denoised anchor as a hold instead of falling back to the older direct pending-`q` correction. Later forecasts retain bounded lambda-space extrapolation; offline replay retains its separate exact-`q` path. Native ER-SDE RNG/stochastic ownership and the no-extra-NFE contract are unchanged.\n\nReal validation used Continuum continuation chunks with Model Preview Override. The previously reproducible confetti on the first post-prefix forecast / third preview step is gone.\n\n'''
if section in text:
    raise SystemExit(0)
if anchor not in text:
    raise SystemExit("README insertion anchor not found")
text = text.replace(anchor, anchor + section, 1)
path.write_text(text, encoding="utf-8")
