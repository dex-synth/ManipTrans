"""Train a residual on mjlab using frozen generalist imitators."""

import argparse

from .training import train_residual, training_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    training_arguments(parser, "residual")
    train_residual(parser.parse_args())


if __name__ == "__main__":
    main()
