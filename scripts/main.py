import os
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")

from scripts.graph_generator import generate_graph


def main():
    generate_graph()


if __name__ == "__main__":
    main()
