"""globsim.globsim: provides entry point main()."""

import argparse
import logging
import sys

from accomatic import accomatic_run

action_dict = {"run": accomatic_run}


logger = logging.getLogger("globsim")


def main():
    mainparser = argparse.ArgumentParser(
        description="Accomatic: Analysis of point-scale simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = mainparser.add_subparsers()
    run = subparsers.add_parser(
        "run", description="Run accomatic based on config file settings."
    )
    mainparser.add_argument(
        "--version", action="version", version=f"Accomatic version {__version__}"
    )

    for parser in [run]:

        parser.add_argument(
            "-f",
            "--config-file",
            default=None,
            type=str,
            required=True,
            dest="f",
            help="Path to accomatic toml configuration file.",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            nargs="?",
            default=logging.INFO,
            const=logging.DEBUG,
            dest="level",
            help="Show detailed output",
        )

    if len(sys.argv) == 1:
        mainparser.print_help(sys.stderr)
        sys.exit(1)

    else:
        args = mainparser.parse_args()
        args.func(args)
