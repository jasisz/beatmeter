# Claude Code Instructions for BeatMeter

## Language
User communicates in Polish. Respond in Polish for conversational messages and English for code/documentation.

## Documentation Maintenance

### RESEARCH.md (docs/RESEARCH.md)
**Always keep up to date.** After any significant experiment, signal change, or benchmark result:
- Update the Abstract if overall accuracy changed
- Update Section 2.2 (Results History) with a new row
- Update Section 2.3 (Per-Category Performance) if category results changed
- Add a new subsection under Section 4 (Experiments) for any new experiment
- Update Section 3 (Literature) if new papers/models were used
- Update References at the end
- Update Section 8 (Future Work) to reflect current priorities

### Experiment Documentation
When running experiments that produce interesting findings (positive or negative):
- Document them in `docs/RESEARCH.md` with full methodology and results
- If the finding is substantial enough (new signal, new model, architectural insight), consider creating a separate doc in `docs/` (e.g., `docs/MERT-EXPERIMENT.md`)
- Always include: motivation, methodology, quantitative results, analysis, conclusion

### Memory Files
Update `.claude/projects/.../memory/MEMORY.md` and topic files after:
- Benchmark results change
- New signals are added/removed
- New lessons learned
- Architecture changes

## Respecting Time & Resources
- **ALWAYS smoke-test with --limit 3 before any long run.** Never launch a multi-hour job without verifying on 3 files first.
- **Parallelize by default.** If tasks are independent (subprocess per file), use --workers from the start, not sequential.
- **METER2800 is the primary benchmark.** Use tuning split (2100) for iteration, test split (700) as hold-out.
- **Don't waste compute on the wrong dataset.** Tune on METER2800 tuning, not on internal fixtures.

## Benchmark Protocol
- Always run `uv run python scripts/eval.py --limit 3 --workers 1` as smoke test after changes to meter.py
- Use `--save` to save new baselines only after confirming improvements
- Use `--verbose` for per-file diagnostics when debugging regressions
- Never skip the gate check for new signals (see methodology.md)

## Signal Integration Protocol
1. Train/implement signal standalone
2. Run orthogonality check (agreement matrix pattern)
3. Gate check: agreement <85%, complementarity ratio >1.5
4. Only then integrate with W_NEW = 0.0, then tune up
5. Document results in RESEARCH.md regardless of outcome (negative results are valuable)
