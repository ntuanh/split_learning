import os
import pika
import pickle
import argparse
import sys
import yaml
import numpy as np

import torch
import requests

import src.Model
import src.Log

from requests.auth import HTTPBasicAuth

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

# parser.add_argument('--topo', type=int, nargs='+', required=True, help='List of client topo, example: --topo 2 3')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

total_clients = config["server"]["clients"]
filename = config["server"]["filename"]
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

num_round = config["server"]["num-round"]
save_parameters = config["server"]["parameters"]["save"]
load_parameters = config["server"]["parameters"]["load"]
validation = config["server"]["validation"]


class Server:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()
        self.num_round = num_round

        self.channel.queue_declare(queue='rpc_queue')
        self.channel.queue_declare('broadcast_queue', durable=False)

        self.total_clients = total_clients
        self.current_clients = [0 for _ in range(len(total_clients))]
        self.register_clients = [0 for _ in range(len(total_clients))]
        self.first_layer_clients = 0
        self.responses = {}  # Save response
        self.list_clients = []

        self.all_model_parameters = [[] for _ in range(len(total_clients))]
        self.all_labels = np.array([])
        self.all_vals = np.array([])

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message
        if (str(client_id), layer_id) not in self.list_clients:
            self.list_clients.append((str(client_id), layer_id))

        if action == "REGISTER":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.notify_clients()
        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            if layer_id == 1:
                self.first_layer_clients += 1
                validate = message["validate"]
                if validate:
                    self.all_labels = np.append(self.all_labels, validate[0])
                    self.all_vals = np.append(self.all_vals, validate[1])

            if self.first_layer_clients == self.total_clients[0]:
                self.first_layer_clients = 0
                src.Log.print_with_color("Received finish training notification", "yellow")

                if self.all_labels.size == self.all_vals.size and self.all_vals.size > 0:
                    same_elements = np.sum(self.all_labels == self.all_vals)
                    total_elements = self.all_vals.size
                    accuracy = same_elements / total_elements
                    src.Log.print_with_color("Inference test: Accuracy: ({:.0f}%)\n".format(100.0 * accuracy), "yellow")

                    self.all_labels = np.array([])
                    self.all_vals = np.array([])

                for _ in range(sum(self.total_clients)):
                    self.send_to_broadcast()
        elif action == "UPDATE":
            data_message = message["message"]
            src.Log.print_with_color(f"[<<<] Received message from client: {data_message}", "blue")
            self.current_clients[layer_id - 1] += 1
            # Save client's model parameters
            if save_parameters:
                model_state_dict = message["parameters"]
                self.all_model_parameters[layer_id - 1].append(model_state_dict)

            # If consumed all client's parameters
            if self.current_clients == self.total_clients:
                src.Log.print_with_color("Collected all parameters.", "yellow")
                if save_parameters:
                    self.avg_all_parameters()
                    self.all_model_parameters = [[] for _ in range(len(total_clients))]
                self.current_clients = [0 for _ in range(len(total_clients))]
                # Test
                if save_parameters and validation:
                    src.Model.test(filename, len(total_clients))
                # Start a new training round
                self.num_round -= 1
                if self.num_round > 0:
                    if save_parameters:
                        self.notify_clients()
                    else:
                        self.notify_clients(register=False)
                else:
                    self.notify_clients(start=False)
                    sys.exit()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True):
        # Send message to clients when consumed all clients
        for (client_id, layer_id) in self.list_clients:
            # Read parameters file
            filepath = f'{filename}_{layer_id}.pth'
            state_dict = None
            if start:
                if load_parameters and register:
                    if os.path.exists(filepath):
                        state_dict = torch.load(filepath, weights_only=False)
                        src.Log.print_with_color("Model loaded successfully.", "green")
                    else:
                        src.Log.print_with_color(f"File {filepath} does not exist.", "yellow")

                src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "parameters": state_dict}
            else:
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
            self.send_to_response(client_id, pickle.dumps(response))

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_channel = self.connection.channel()
        reply_queue_name = f'reply_{client_id}'
        reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def send_to_broadcast(self):
        broadcast_channel = self.connection.channel()
        broadcast_queue_name = 'broadcast_queue'
        broadcast_channel.queue_declare(broadcast_queue_name, durable=False)

        message = pickle.dumps({"action": "STOP",
                                "message": "Stop training and please send your parameters",
                                "parameters": None})
        broadcast_channel.basic_publish(
            exchange='',
            routing_key=broadcast_queue_name,
            body=message
        )

    def avg_all_parameters(self):
        # Average all client parameters
        for layer, state_dicts in enumerate(self.all_model_parameters):
            avg_state_dict = {}
            num_models = len(state_dicts)

            for key in state_dicts[0].keys():
                if state_dicts[0][key].dtype == torch.long:
                    avg_state_dict[key] = state_dicts[0][key].float()
                else:
                    avg_state_dict[key] = state_dicts[0][key].clone()

                for i in range(1, num_models):
                    if state_dicts[i][key].dtype == torch.long:
                        avg_state_dict[key] += state_dicts[i][key].float()
                    else:
                        avg_state_dict[key] += state_dicts[i][key]

                avg_state_dict[key] /= num_models

                if state_dicts[0][key].dtype == torch.long:
                    avg_state_dict[key] = avg_state_dict[key].long()

            # Save to files
            torch.save(avg_state_dict, f'{filename}_{layer + 1}.pth')


def delete_old_queues():
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' purged.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

        connection.close()
        return True
    else:
        src.Log.print_with_color(f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
        return False


if __name__ == "__main__":
    delete_old_queues()
    server = Server()
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
