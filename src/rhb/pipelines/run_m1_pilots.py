"""One-command execution of both M1 pilot fits and their decision audits.

Run from the repository root:
    python -m rhb.pipelines.run_m1_pilots

Requires the existing modules run_m1a_pilot.py, run_m1b_pilot.py and their source data.
Existing pilot outputs are overwritten by re-fitting; save/archive them first if needed.
"""
from __future__ import annotations


def main() -> None:
    from rhb.pipelines.run_m1a_pilot import main as run_m1a
    from rhb.pipelines.run_m1b_pilot import main as run_m1b
    from rhb.pipelines.audit_m1_pilots import main as audit

    run_m1a()
    run_m1b()
    audit()


if __name__ == "__main__":
    main()
