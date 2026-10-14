"""Verify legacy merge and declared-base delta composition of two PR overlays.

All writes occur in temporary linked worktrees. The source checkout is read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


def git(repo, *args, input_data=None):
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Overlay validation",
            "-c",
            "user.email=overlay-validation@local.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        input=input_data,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def apply_delta(repo, declared_base, head):
    branch_point = git(repo, "merge-base", declared_base, head).decode().strip()
    patch = git(
        repo,
        "diff",
        "--binary",
        "--full-index",
        "--find-renames",
        branch_point,
        head,
        "--",
    )
    if patch:
        git(repo, "apply", "--3way", "--index", input_data=patch)
    git(
        repo,
        "commit",
        "--no-verify",
        "--allow-empty",
        "-m",
        "Synthetic overlay validation",
    )


def verify(repo, base, parent, head):
    repo = Path(repo).resolve()
    base, parent, head = [
        git(repo, "rev-parse", ref).decode().strip() for ref in (base, parent, head)
    ]
    git(repo, "merge-base", "--is-ancestor", parent, head)
    expected = git(repo, "rev-parse", f"{head}^{{tree}}").strip()
    for mode in ("legacy_merge", "declared_base_delta"):
        with tempfile.TemporaryDirectory(prefix="spectrum-overlay-") as directory:
            worktree = Path(directory) / "checkout"
            git(repo, "worktree", "add", "--detach", str(worktree), base)
            try:
                if mode == "legacy_merge":
                    for incoming in (parent, head):
                        git(
                            worktree,
                            "merge",
                            "--no-ff",
                            "--no-edit",
                            "-m",
                            "Overlay validation",
                            incoming,
                        )
                else:
                    apply_delta(worktree, base, parent)
                    apply_delta(worktree, parent, head)
                actual = git(worktree, "rev-parse", "HEAD^{tree}").strip()
                if actual != expected:
                    raise RuntimeError(
                        f"{mode}: composed tree differs from the candidate"
                    )
                print(f"{mode}: clean; composed tree equals candidate {head}")
            finally:
                git(repo, "worktree", "remove", "--force", str(worktree))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    verify(args.repo, args.base, args.parent, args.head)


if __name__ == "__main__":
    main()
