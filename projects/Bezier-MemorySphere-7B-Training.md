# Bezier-MemorySphere-7B-Training

`Bezier-MemorySphere-7B-Training` · Memory systems

Run a substantial time-bounded local training experiment from scratch with fuller token supervision and a broader coding/action corpus.

## Saved status

Training completed; final task panels incomplete; no working coding-agent capability demonstrated.

## Work completed or attempted

- Built a 7,405,637,063-parameter variant with an own 16K tokenizer, full sampled code-window and action-token targets, staged code/agent mixing, and stochastic BF16 Adafactor updates.
- Prepared 32000 Python files (42.6M stored training tokens), 6575 coding tasks, and 48118 supervised actions; froze validation/test panels and retained provenance/filtering records.
- Completed the ten-hour run and a later read-only checkpoint/generation diagnostic; retained completion, token losses, partial task panels, model shards, optimizer/RNG state, and traces.

## Findings

**Measured.** Training completed on 2026-11-24 at 11:52:48 UTC (06:52:48 America/New_York), after 22,042 updates and 4,774,503 supervised tokens. The best/evaluated checkpoint was u0022042. This supersedes the README/handoff's live or underway description.

[S103](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/COMPLETE.json) · [S104](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/status.json)

**Measured.** Final evaluation timed out before both 96-task panels finished: eight visits scored 0/60 and adaptive scored 0/40, with zero passing code or finished workflows. These are incomplete panels, not 0/96 results.

[S103](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/COMPLETE.json)

**Measured.** Validation code NLL improved from 9.7171 to 5.9733 and agent NLL from 9.7395 to 2.4168; agent token accuracy reached 47.20%. However, all 180 recorded eight-visit actions and all 116 adaptive actions were invalid, with no successful edits.

[S105](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/checkin_2026-11-24.json)

**Measured.** The later check-in verified all 27 checkpoint-file SHA-256 checks and frozen source matches. Full-context decoding on two validation prompts also generated malformed repetitive output; it differed from cached decoding. No training was restarted by the check-in.

[S105](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/checkin_2026-11-24.json) · [S106](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/generation_diagnostic.json)

**Measured.** The run recorded 30.097 GiB peak allocated tensors, 7696 distinct code records and 9960 distinct agent records. Stored corpus tokens are not equivalent to the 4.77M actually supervised tokens.

[S103](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/COMPLETE.json)

## Limits and open questions

- Training-budget completion does not establish working agentic coding; held-out scoring was partial and yielded no successes.
- Single model/run, limited training exposure, no matched from-scratch geometry controls, and incomplete semantic decontamination.
- Cached versus full-context outputs differ; diagnostic malformed output in both paths does not identify a single failure cause.
- Historical status keep_awake/pid fields are saved metadata, not proof a worker is still running.

## Recommended next steps

- Recommendation: Prioritize valid tool generation and cached/full-context consistency diagnostics, then complete fixed held-out panels before any longer training claim.
- Recommendation: Update project-facing status prose to the completion/check-in evidence while preserving original records.

## Sources

- [Training design and data](../evidence/Bezier-MemorySphere-7B-Training/README.md)
- [Authoritative completion and partial results](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/COMPLETE.json)
- [Last saved status](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/status.json)
- [Post-run integrity and action audit](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/checkin_2026-11-24.json)
- [Cached versus full decoding diagnostic](../evidence/Bezier-MemorySphere-7B-Training/runs/coding_agent_overnight_v1/generation_diagnostic.json)

## Related projects

- [Bezier-MemorySphere-7B](Bezier-MemorySphere-7B.md)
- [Bezier-MemorySphere](Bezier-MemorySphere.md)

Latest reviewed follow-up: 2026-11-24 21:46 UTC (16:46 America/New_York). Main run completed 2026-11-24 11:52 UTC. This index does not assert current process liveness.

[Back to the project directory](README.md)
