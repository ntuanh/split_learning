import pika
import pickle
import requests
from requests.auth import HTTPBasicAuth

total_clients = [1, 1]
file_path = "./test.pt"
address = "192.168.101.234"
username = "dai"
password = "dai"


class Server:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue='rpc_queue')
        self.channel.queue_declare('broadcast_queue', durable=False)

        self.total_clients = total_clients
        self.current_clients = [0 for _ in range(len(total_clients))]
        self.first_layer_clients = 0
        self.responses = {}  # Save response

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        print(f"Server is waiting for {self.total_clients} clients.")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        print(message)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        print(f"Received message from client: {message}")

        if action == "REGISTER":
            # Save messages from clients
            self.responses[routing_key] = message
            self.current_clients[layer_id - 1] += 1

            # If consumed all clients
            if self.current_clients == self.total_clients:
                print("All clients are connected. Sending notifications.")
                self.notify_clients(ch)
                self.current_clients = [0 for _ in range(len(total_clients))]
        if action == "NOTIFY":
            if layer_id == 1:
                self.first_layer_clients += 1

            if self.first_layer_clients == self.total_clients[0]:
                print("Received finish training notification")
                self.stop_training_round(ch)
                for _ in range(sum(self.total_clients[1:])):
                    self.send_to_broadcast()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, channel):
        # Send message to clients when consumed all clients
        for routing_key in self.responses:
            response = {"action": "START",
                        "message": "Server accepted are connection!",
                        "parameters": None}
            # Read parameters file
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    response["parameters"] = content
                    # TODO: Send with corresponding layers
            except FileNotFoundError:
                print(f"File {file_path} does not exist.")

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
            if queue_name.startswith("amq.gen-"):
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
