import argparse

from src.Server import Server
import src.Log

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

# parser.add_argument('--topo', type=int, nargs='+', required=True, help='List of client topo, example: --topo 2 3')

args = parser.parse_args()


if __name__ == "__main__":
    server = Server('config.yaml')
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
