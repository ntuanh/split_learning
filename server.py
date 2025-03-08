import argparse
import sys
import signal
from src.Server import Server
from src.Utils import delete_old_queues
import src.Log

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

# parser.add_argument('--topo', type=int, nargs='+', required=True, help='List of client topo, example: --topo 2 3')

args = parser.parse_args()


def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues('192.168.101.91','dai', 'dai')
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    server = Server('config.yaml')
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
