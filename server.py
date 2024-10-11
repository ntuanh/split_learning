import pika
import pickle
import os

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


if __name__ == "__main__":
    server = Server()
    server.start()
    print("Ok, ready!")
