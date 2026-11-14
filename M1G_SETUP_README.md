# M1-G package | application instructions

This ZIP is an **additive patch for the scientific** `resilient-housing-bayes` repo, not `Habnetic/data`.

1. Work on a new Git branch from the currently frozen M1 code and cleanly review `git status`.
2. Expand the ZIP **into `resilient-housing-bayes`**, not its parent. The ZIP deliberately contains no root `README.md` to avoid clobbering your repository's introduction.
3. Verify existing `src/rhb/pipelines/run_m1c.py`, `src/rhb/experiments/m1f_population_sampling.py`, and both model functions. No copies of these frozen files are included here.
4. Run `pytest tests/test_m1g.py -q` and the city input `--check` before generating any scenarios.
5. Commit the new source/config/docs **before inference** to capture a truthful code SHA in run manifests.
6. Run the smoke experiments, review flags and summaries, then run the bounded full studies.
7. Archive the generated outputs separately and **only tag after scientific review**.

See `docs/phase4/M1_G_Protocol_and_Runbook.md` for full design, exact commands, scientific limitations and outputs.

This ZIP was built against the frozen M1-C and M1-F source artifacts retained in Habnetic's project files, not against a live Windows checkout. The read-only source/data audit is therefore mandatory.
