import time
import pickle
import pika


class RpcClient:
    def __init__(self, client_id, layer_id, model, address, username, password, train_func, train_data=None):
        self.model = model
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.train_data = train_data

        self.channel = None
        self.connection = None
        self.response = None

        self.connect()

    def wait_response(self):
        status = True
        while status:
            credentials = pika.PlainCredentials(self.username, self.password)
            reply_connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
            reply_channel = reply_connection.channel()
            reply_queue_name = f'reply_{self.client_id}'
            reply_channel.queue_declare(reply_queue_name, durable=False)
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                # print(f"Received message from server {received_data}")
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        print(f"Client received: {self.response['message']}")
        action = self.response["action"]
        parameters = self.response["parameters"]

        if action == "START":
            # Read parameters and load to model
            if parameters:
                self.model.to("cpu")
                self.model.load_state_dict(parameters)
            # Start training
            if self.layer_id == 1:
                self.train_func(self.train_data)
            else:
                self.train_func()
            # Stop training, then send parameters to server
            self.model.to("cpu")
            model_state_dict = self.model.state_dict()
            data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                    "message": "Send parameters to Server", "parameters": model_state_dict}
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
