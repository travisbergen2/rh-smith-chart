# The RH Smith Chart

**An instrument, not a proof.** This project proves nothing about the Riemann
Hypothesis and displays no evidence for or against it.

The classical Smith chart folds the right half-plane of positive-real
impedances into the unit disk via `Γ = (Z−1)/(Z+1)`, making passivity readable
as geometry. Li's map `z = 1 − 1/s` performs the identical fold on the critical
strip: RH ⇔ every nontrivial zero's image lies on the unit circle. The two
circle-side positivity gauges of the IMM program — Li coefficients λ_n and
Paper 14's A(m) Toeplitz truncations — are rendered on one instrument panel,
with a synthetic planted violation to show what failure would look like.

## Files

| file | role |
|---|---|
| `core.py` | maps, λ_n (zero-side, truncated), A(m)/B-tent (Paper 14, erratum convention), Toeplitz μ_min. Epistemic labels in docstrings. |
| `test_core.py` | 9 unit tests (all passing). Run `python3 test_core.py`. |
| `make_chart_data.py` | generates `data/chart_data.json`. All committed parameters live here. |
| `build_html.py` + `chart_template.html` | assemble the self-contained interactive chart `rh_smith_chart.html`. |
| `rh_smith_note.tex` / `.pdf` | the companion note (IMM Open Series draft, number unassigned). |
| `data/zeros1.txt` | Odlyzko table, first 100,000 zeros (not committed to git; download from Odlyzko's zeta tables). |

## Reproduce

```bash
pip install numpy scipy mpmath
curl -o data/zeros1.txt https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1
python3 test_core.py          # 9/9 PASS expected
python3 make_chart_data.py    # ~11 s
python3 build_html.py         # -> rh_smith_chart.html
```

## Committed parameters

- zeros: first 100,000 (T ≈ 74,921), Odlyzko
- δ = 0.05 (Paper 14 primary), depth n, M ≤ 445 (E-LI-1 gate G2's binding D_max)
- planted demo: quadruple 1/2 ± 0.1 ± i·98.831194 replacing the on-line pair
  at that height — E-LI-1's one honest detection cell

## Cross-validation

- λ₁ (truncated, no tail) = 0.0230736 vs published 0.0230957 — gap is the
  omitted tail integral, deliberate and labeled
- μ_min(T_445) = 4.84e−6 > 0 on the true zero set
- planted-violation detection at **M = 272**, matching E-LI-1's independent
  endpoint run exactly

## Epistemic grades

Dictionary rows are graded [T]/[A]/[Open] per IMM Paper 0B conventions; see the
note. The half-plane→disk positivity bridge (Li 1997; de Branges; Suzuki
arXiv:1204.1827) is standard mathematics. The contribution here is assembly and
instrumentation only.
