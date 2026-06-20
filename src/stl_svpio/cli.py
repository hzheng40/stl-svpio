from __future__ import annotations

import argparse

from stl_svpio.scripts.reproduce_figure3 import main as figure3_main
from stl_svpio.scripts.reproduce_nonlinear import main as nonlinear_main
from stl_svpio.scripts.reproduce_table1 import main as table1_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STL-SVPIO paper reproduction command.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("table1", help="Reproduce Table I / Figure 2 reach-avoid runs.")
    sub.add_parser("figure3", help="Reproduce Figure 3 point-mass benchmark runs.")
    sub.add_parser("nonlinear", help="Show or launch Panda/Half-Cheetah commands.")
    args, rest = parser.parse_known_args(argv)
    if args.command == "table1":
        table1_main(rest)
    elif args.command == "figure3":
        figure3_main(rest)
    elif args.command == "nonlinear":
        nonlinear_main(rest)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

