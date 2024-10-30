import pika
import uuid
import pickle
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

import src.Log
from src.RpcClient import RpcClient
from Model import ModelPart2

parser = argparse.ArgumentParser(description="Split learning framework")
# parser.add_argument('--id', type=int, required=True, help='ID of client')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

layer_id = 2
client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
control_count = config["learning"]["control-count"]

device = None

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print(f"Using device: CPU")

model = ModelPart2()
optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
channel = connection.channel()


def send_intermediate_output(data_id, output, labels, trace):
    forward_queue_name = f'intermediate_queue_{layer_id}'
    channel.queue_declare(forward_queue_name, durable=False)
    trace.append(client_id)

    message = pickle.dumps({"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": trace})

    channel.basic_publish(
        exchange='',
        routing_key=forward_queue_name,
        body=message
    )


def send_gradient(data_id, gradient, trace):
    to_client_id = trace[-1]
    trace.pop(-1)
    backward_queue_name = f'gradient_queue_{layer_id - 1}_{to_client_id}'
    channel.queue_declare(queue=backward_queue_name, durable=False)

    message = pickle.dumps({"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace})

    channel.basic_publish(
        exchange='',
        routing_key=backward_queue_name,
        body=message
    )


def train_on_device():
    forward_queue_name = f'intermediate_queue_{layer_id - 1}'
    backward_queue_name = f'gradient_queue_{layer_id}_{client_id}'
    channel.queue_declare(queue=forward_queue_name, durable=False)
    channel.queue_declare(queue=backward_queue_name, durable=False)
    channel.basic_qos(prefetch_count=10)
    data_store = {}
    print('Waiting for intermediate output. To exit press CTRL+C')
    model.to(device)
    while True:
        # Training model
        model.train()
        optimizer.zero_grad()
        # Process gradient
        method_frame, header_frame, body = channel.basic_get(queue=backward_queue_name, auto_ack=True)
        if method_frame and body:
            received_data = pickle.loads(body)
            gradient_numpy = received_data["data"]
            gradient = torch.tensor(gradient_numpy).to(device)
            trace = received_data["trace"]
            data_id = received_data["data_id"]

            data_input = data_store.pop(data_id)
            output = model(data_input)
            data_input.retain_grad()
            output.backward(gradient=gradient, retain_graph=True)
            optimizer.step()

            gradient = data_input.grad
            send_gradient(data_id, gradient, trace)
        else:
            method_frame, header_frame, body = channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]
                data_id = received_data["data_id"]
                labels = received_data["label"].to(device)

                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(device)
                data_store[data_id] = intermediate_output

                output = model(intermediate_output)
                output = output.detach().requires_grad_(True)

                send_intermediate_output(data_id, output, labels, trace)
                # speed control
                if len(data_store) > control_count:
                    continue
        # Check training process
        if method_frame is None:
            broadcast_queue_name = 'broadcast_queue'
            method_frame, header_frame, body = channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                break


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": layer_id, "message": "Hello from Client!"}
    client = RpcClient(client_id, layer_id, model, address, username, password, train_on_device)
    client.send_to_server(data)
    client.wait_response()
