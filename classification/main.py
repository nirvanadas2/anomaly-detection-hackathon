"""
Usage:
    python -m classification.main
"""

from . import classify, evaluate


def main():
    classify.run(verbose=True)
    print()
    evaluate.main()


if __name__ == "__main__":
    main()
