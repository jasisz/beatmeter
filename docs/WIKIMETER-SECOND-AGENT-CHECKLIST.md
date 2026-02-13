# WIKIMETER Second-Agent Review Checklist

Use this as an independent audit prompt for a second AI agent.

## Goal

Review and validate the curation pipeline:

1. `scripts/setup/collect_wikimeter_candidates.py`
2. `scripts/setup/build_wikimeter_review_queue.py`
3. `scripts/setup/verify_wikimeter_sources.py`
4. `scripts/setup/wikimeter_agent_swarm.py`
5. `scripts/setup/merge_wikimeter_reviewed.py`
6. `scripts/setup/wikimeter_tools.py`

Focus on correctness, data quality, and silent failure risks.

## Required checks

1. Confirm all scripts run with `--help`.
2. Confirm `python3 -m py_compile` succeeds for all files above.
3. Validate JSON schemas:
   - candidates payload shape
   - review queue payload shape
   - verification payload (`verification.evidence`, `suggested_status`, `confidence`)
4. Verify dedup behavior:
   - duplicate source handling (`type+url` / `youtube:video_id`)
   - duplicate song handling
   - replace-vs-skip merge behavior
5. Verify verification logic:
   - sources include YouTube metadata, Wikipedia, and non-Wikipedia web pages
   - rare meters (`9/x`, `11/x`) and poly stay manual (`pending`)
   - conflicting evidence can suggest `rejected`
6. Verify swarm logic:
   - `scout` and `skeptic` produce separate evidence sets
   - `arbiter` decision uses both agents (not just one)
   - `--apply-consensus` writes status and review note
7. Check robustness:
   - network failure path does not crash whole run
   - missing `yt-dlp` produces readable errors
   - malformed entries are skipped with counts/logs

## Minimal smoke test commands

```bash
python3 -m py_compile \
  scripts/setup/wikimeter_tools.py \
  scripts/setup/collect_wikimeter_candidates.py \
  scripts/setup/build_wikimeter_review_queue.py \
  scripts/setup/verify_wikimeter_sources.py \
  scripts/setup/wikimeter_agent_swarm.py \
  scripts/setup/merge_wikimeter_reviewed.py

python3 scripts/setup/collect_wikimeter_candidates.py --help
python3 scripts/setup/build_wikimeter_review_queue.py --help
python3 scripts/setup/verify_wikimeter_sources.py --help
python3 scripts/setup/wikimeter_agent_swarm.py --help
python3 scripts/setup/merge_wikimeter_reviewed.py --help
```

## Review output format

Return findings in this order:

1. Critical bugs (with file + line reference)
2. Data quality risks
3. Operational risks (cost/time/rate-limit)
4. Suggested fixes (small, concrete patches first)

If no major issues are found, state that explicitly and list residual risks.
