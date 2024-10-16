import os
import pika
import pickle
import argparse
import sys
import yaml

import torch

import requests
from requests.auth import HTTPBasicAuth
import src.Model

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
        self.first_layer_clients = 0
        self.responses = {}  # Save response

        self.all_model_parameters = [[] for _ in range(len(total_clients))]

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        print(f"Server is waiting for {self.total_clients} clients.")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            print(f"Received message from client: {message}")
            # Save messages from clients
            self.current_clients[layer_id - 1] += 1

            # If consumed all clients
            if self.current_clients == self.total_clients:
                print("All clients are connected. Sending notifications.")
                self.notify_clients(ch)
                self.current_clients = [0 for _ in range(len(total_clients))]
        elif action == "NOTIFY":
            print(f"Received message from client: {message}")
            if layer_id == 1:
                self.first_layer_clients += 1

            if self.first_layer_clients == self.total_clients[0]:
                self.first_layer_clients = 0
                print("Received finish training notification")
                self.stop_training_round(ch)
                for _ in range(sum(self.total_clients[1:])):
                    self.send_to_broadcast()
        elif action == "UPDATE":
            # Save client's model parameters
            model_state_dict = message["parameters"]
            self.current_clients[layer_id - 1] += 1
            self.all_model_parameters[layer_id - 1].append(model_state_dict)

            # If consumed all client's parameters
            if self.current_clients == self.total_clients:
                print("Collected all parameters.")
                self.avg_all_parameters()
                self.current_clients = [0 for _ in range(len(total_clients))]
                self.all_model_parameters = [[] for _ in range(len(total_clients))]
                # Test
                if validation:
                    src.Model.test(filename, len(total_clients))
                # Start a new training round
                self.num_round -= 1
                if self.num_round > 0:
                    self.notify_clients(ch)
                else:
                    self.notify_clients(ch, start=False)
                    sys.exit()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, channel, start=True):
        # Send message to clients when consumed all clients
        for routing_key in self.responses:
            layer = self.responses[routing_key]["layer_id"]
            # Read parameters file
            filepath = f'{filename}_{layer}.pth'
            state_dict = None
            if start:
                if os.path.exists(filepath):
                    state_dict = torch.load(filepath, weights_only=False)
                    print("Model loaded successfully.")
                else:
                    print(f"File {filepath} does not exist.")

                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "parameters": state_dict}
            else:
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": state_dict}

            channel.basic_publish(exchange='',
                                  routing_key=routing_key,
                                  properties=pika.BasicProperties(),
                                  body=pickle.dumps(response))
            print(f"Sent notification to client {routing_key}")

    def stop_training_round(self, channel):
        # Send message to clients when consumed all clients
        for routing_key in self.responses:
            response = {"action": "STOP",
                        "message": "Stop training and please send your parameters",
                        "parameters": None}

            channel.basic_publish(exchange='',
                                  routing_key=routing_key,
                                  properties=pika.BasicProperties(),
                                  body=pickle.dumps(response))
            print(f"Send stop training request to clients {routing_key}")

    def start(self):
        self.channel.start_consuming()

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
            if queue_name.startswith("rpc_callback") or queue_name.startswith(
                    "intermediate_queue") or queue_name.startswith("gradient_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    print(f"Queue '{queue_name}' deleted.")
                except Exception as e:
                    print(f"Failed to delete queue '{queue_name}': {e}")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    print(f"Queue '{queue_name}' deleted.")
                except Exception as e:
                    print(f"Failed to purge queue '{queue_name}': {e}")

        connection.close()
        return True
    else:
        print(f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}")
        return False


if __name__ == "__main__":
    server = Server()
    delete_old_queues()
    server.start()
    print("Ok, ready!")
