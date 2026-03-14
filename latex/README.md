# latex/ — Paper Source

LaTeX source for the NeurIPS 2026 submission.

## Files

| File | Description |
|------|-------------|
| `neurips_2025_main.tex` | Main paper (introduction, method, related work, conclusion) |
| `Experiment1.tex` | Experiment section 1 (synthetic datasets, main results) |
| `Experiment2.tex` | Experiment section 2 (real-world datasets, ablations) |

## Building the PDF

```bash
cd latex/
pdflatex neurips_2025_main.tex
bibtex neurips_2025_main
pdflatex neurips_2025_main.tex
pdflatex neurips_2025_main.tex
```

Result tables from `results/` are referenced in `Experiment1.tex` and `Experiment2.tex`.
