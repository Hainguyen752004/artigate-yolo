# Paper 1 manuscript (IEEE Access): ArtiGate-YOLO

Draft of the main manuscript and the Supplementary Material. **No result in these files is real.**
Every number that the experiments must produce is written as `\res` or `\TBD{...}` and shows up in
red in the PDF. The only numbers that are not marked are design constants read from the code on this
branch, and the parameter and GFLOP counts measured on the fused models (640 x 640, nc = 1).

## Files

| File | Content |
|---|---|
| `main.tex` | Main manuscript (IEEEtran journal class); inputs `sections/*.tex` |
| `supplementary.tex` | Supplementary Material (article class; sections, tables, figures and equations numbered S1, S2, ...) |
| `macros.tex` | `\TBD`, `\res`, `\placeholderfig` and other shared macros (both documents) |
| `refs.bib` | References with a check status above each entry |
| `figures/overview.tex` | Fig. 1, TikZ (architecture and AGA block) |
| `figures/synthetic_artifacts.jpg` | Fig. 2, from `scripts/figure_synthetic_artifacts.py` (placeholder on a drawn lesion) |

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
```

Overleaf: upload the zip, set the main document to `main.tex` (or `supplementary.tex`); the default
pdfLaTeX compiler works. Needed packages are in TeX Live (IEEEtran, tikz, booktabs, tabularx, float,
tocloft).

## Moving into the official IEEE Access template

`main.tex` uses the standard `IEEEtran` journal class so the draft builds anywhere. For submission:

1. Open the official "IEEE Access LaTeX template" (Overleaf gallery, or the template on the IEEE
   Access "Preparing your article" page) and keep its class file and front matter (`\history`,
   `\doi`, author `\address`, `\corresp`, `\tfootnote`, author biographies with photos, `\EOD`).
2. Copy `macros.tex`, `refs.bib`, `sections/`, `figures/` into that project and `\input` the
   sections in the same order as in `main.tex`. Keep `\input{macros}` in the preamble.
3. The abstract and keywords environments of the template replace those in `sections/abstract.tex`.
   IEEE Access asks for a single-paragraph abstract of 150 to 250 words without citations.
4. Check the reference style and the figure/table caption style produced by the template class.
5. Upload `supplementary.pdf` as a separate supplementary file.

## Before submission (search for `\TBD`, `\res`, `VERIFY`, `TODO`)

- [ ] Fill every `\res` and `\TBD` from `runs/` with the scripts below and rewrite each results
      paragraph around what the numbers show, including the comparisons that A4 loses.
- [ ] Decide and record **before scoring the test set**: the primary mask rule (top-1 or union) and
      the non-inferiority margin delta in criterion (ii) (draft: 0.5 Dice points).
- [ ] Check every reference marked `% VERIFY` against the publisher or arXiv page (authors, title,
      venue, pages, DOI) and fill the `% TODO` entry for the ICEEIS paper (`iceeis2026own`).
- [ ] Regenerate Fig. 2 on an ISIC 2018 *training* image:
      `python scripts/figure_synthetic_artifacts.py --image <ISIC_xxx.jpg> --mask <ISIC_xxx_segmentation.png>`.
- [ ] Replace the placeholder figures (severity sweep, qualitative results, training histories,
      augmented batch).
- [ ] Update the AI-use statement in `sections/declarations.tex` to what was actually done.
- [ ] Check overlap with the ICEEIS paper (IEEE Access limits for extended conference papers) and
      state the relation in the cover letter.

## Where each table comes from

| Table (main / supplementary) | Script |
|---|---|
| Main comparison, ablation, S5-S7 | `scripts/eval.py` per run and seed, then `scripts/compare_family.py` (seed averaging, paired bootstrap, Wilcoxon, Holm; one call = one family) |
| Artifact strata, S9-S10 | `scripts/compare_family.py --strata <test flags> --subsets any_artifact,with_hair,with_ruler,with_marker,with_dark_corner,no_artifact` |
| Test-set artifact labels, kappa (S8) | `scripts/label_artifacts.py` |
| Severity sweep (Fig. 3, S11) | `scripts/make_severity_sets.py`, then `scripts/eval.py` on each folder |
| Artifact head (S12-S13) | `scripts/eval_artifact_head.py` |
| Cost (S15) | `scripts/benchmark.py --half` on the target GPU; params/GFLOPs from the Ultralytics model info of the fused model |
| Leakage (S20) | `scripts/check_leakage.py` |
| Qualitative figure | `scripts/viz_artifact_maps.py` |
| Augmented batch (Fig. S2) | `scripts/viz_train_batch.py` |

Families for Holm (as pre-specified in the Statistical Analysis section): primary = A4 vs A1 on the
artifact strata; ablation = each row vs A4 on the full test set; external baselines vs A4.
