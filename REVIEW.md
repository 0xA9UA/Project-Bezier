# Second-pass research and privacy review

Reviewed November 25, 2026. The starting point was publication commit `b13b3478f8d095efe2b6bf1dbb7b67b986e55407`: 282 files, 26 project folders, 137 findings and 247 source references. Every staged file was matched to the published Git blob before review. This review does not rerun training or independently reproduce the models.

## Research usefulness

This is useful as an exploratory record of controls, engineering tradeoffs, negative results and specific leads for replication. It does not establish a generally superior architecture or novel scientific result. [Start with the practical reader guide](READ_THIS_FIRST.md).

The reviewed headline measurements agree with their cited saved artifacts. A recount of v8's 768 test records reproduced 490 learned, 485 fixed, 481 shuffled and 495 dense correct answers, with 17 rescues and 12 spoils versus fixed routing. The records represent 192 program groups, not 768 independent programs. [V8 test artifact](evidence/bezier_nested_llm_7b_v8/runs/pilot/test_evaluation.json)

The main positive steering result needed a clearer denominator: v4's 512 answer-prefix cases came from 32 program files and one training seed. Accuracy rose from 52.73% to 57.62% when redirects were enabled in the same checkpoint; the separately trained clean model reached 57.03%. The 4.89-point gain is against disabled redirects, not that clean control. The README, project notes, atlas and PDF now make this boundary explicit. [V4 report](evidence/bezier_nested_core_v4/runs/input_preserving/report.md) · [Raw suite](evidence/bezier_nested_core_v4/runs/input_preserving/suite.json)

Historical source documents describe a larger workspace. Their claims about local code, data and weights must not be read as a promise that those artifacts are distributed here. Scope notices now appear on the copied Markdown and HTML reports. The public snapshot remains inspectable evidence, not a runnable release of all 26 projects. Original source links can still refer to omitted material. [Publication scope](evidence/PUBLICATION_SCOPE.md)

## Privacy checks

The review scanned every published text file, parsed JSON and embedded HTML data, examined selected prompts and generated code in context, inspected all figures and their metadata, checked PDF actions and attachments, and reviewed the two reachable Git commits and the sole branch at the starting revision.

No credentials, private contact details, private home-directory paths, internal network addresses, signed authentication URLs, private chat records or identifiable client documents were confirmed in that review. Contextual review is not proof that every possible sensitive string has been recognized.

One chart's PNG text metadata retained workspace file paths. Those were project-local paths, not personal home directories, network locations or credentials. PNG text, EXIF and time metadata were removed from the published figures; their decoded pixels are unchanged. The source manifest records the new published hashes.

The PDF contains no attachments, JavaScript or executable launch actions; its citations use HTTPS GitHub URLs. The HTML reports contain no detected automatic external network calls or third-party scripts. Model-weight and dataset binaries are not included. Hardware/software versions, dates and source fingerprints remain as relevant research provenance. Git commits use the account's public display name and GitHub noreply address.

These corrections apply to the current publication. Earlier Git objects can remain retrievable; this review does not claim that GitHub history, caches or external copies were erased. No history rewrite was performed.
