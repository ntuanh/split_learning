import time
import pickle
import pika
import src.Log
from torch import nn

from src.Model import VGG16

full_model = VGG16()


class RpcClient:
    def __init__(self, client_id, layer_id, address, username, password, train_func, train_data=None, test_data=None):
        # self.model = model
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.train_data = train_data
        self.test_data = test_data

        self.channel = None
        self.connection = None
        self.response = None
        self.model = None
        self.connect()

    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        parameters = self.response["parameters"]

        if action == "START":
            cut_layers = self.response['layers']
            # Read parameters and load to model
            if cut_layers:
                a = cut_layers[0]
                b = cut_layers[1]
                if b == -1:
                    self.model = nn.Sequential(*nn.ModuleList(full_model.children())[a:])
                else:
                    self.model = nn.Sequential(*nn.ModuleList(full_model.children())[a:b])

            if parameters:
                self.model.to("cpu")
                self.model.load_state_dict(parameters)

            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]

            # Start training
            if self.layer_id == 1:
                self.train_func(self.model, self.train_data, self.test_data)
            else:
                self.train_func(self.model)
            # Stop training, then send parameters to server
            self.model.to("cpu")
            model_state_dict = self.model.state_dict()
            data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
