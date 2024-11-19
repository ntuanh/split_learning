import pika
import uuid
import argparse
import yaml

import torch

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler


parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--num_layers', type=int, required=True, help='Number of split layers')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

layer_id = args["layer_id"]
num_layers = args["num_layers"]

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

device = None

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print(f"Using device: CPU")

credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
channel = connection.channel()


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": layer_id, "message": "Hello from Client!"}
    scheduler = Scheduler(client_id, layer_id, channel, device, num_layers)
    client = RpcClient(client_id, layer_id, address, username, password, scheduler.train_on_device)
    client.send_to_server(data)
    client.wait_response()
