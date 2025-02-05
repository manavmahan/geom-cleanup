"""Utility functions."""

import subprocess


def get_git_info():
    try:
        # Get the current branch name
        branch_name = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )

        # Get the current commit ID
        commit_id = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.STDOUT
            )
            .strip()
            .decode("utf-8")
        )

        return branch_name, commit_id
    except subprocess.CalledProcessError as e:
        print(f"Error while fetching git information: {e.output.decode('utf-8')}")
        return None, None
