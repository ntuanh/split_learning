import uuid
import pickle
import pika
import time


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
        self.callback_queue = f"rpc_callback_{uuid.uuid4()}"

        self.response = None
        self.corr_id = None

        self.connect()

    def on_response(self, ch, method, props, body):
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
            self.send_to_server(data, wait=False)

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()
        # Declare an exclusive queue
        self.channel.queue_declare(queue=self.callback_queue, exclusive=True)
        self.channel.basic_consume(queue=self.callback_queue,
                                   on_message_callback=self.on_response,
                                   auto_ack=True)

    def reconnect(self):
        print("Reconnecting to RabbitMQ...")
        try:
            if self.connection and self.connection.is_open:
                self.connection.close()

            self.connect()  # Call connect to re-establish connection

            # Ensure the exclusive queue is re-declared
            self.channel.queue_declare(queue=self.callback_queue, exclusive=True)
            print("Reconnection successful!")

        except Exception as e:
            print(f"Failed to reconnect: {e}")
            raise

    def send_to_server(self, message, wait=True):
        self.response = None
        self.corr_id = str(uuid.uuid4())

        attempts = 0
        max_retries = 5
        retry_delay = 1

        while attempts < max_retries:
            try:
                self.channel.basic_publish(exchange='',
                                           routing_key='rpc_queue',
                                           properties=pika.BasicProperties(
                                               reply_to=self.callback_queue,
                                               correlation_id=self.corr_id),
                                           body=pickle.dumps(message))
                print(f"Message sent. Attempt {attempts + 1}")

                if wait:
                    while self.response is None:
                        self.connection.process_data_events()

                return self.response

            except (pika.exceptions.ConnectionClosed, pika.exceptions.StreamLostError) as e:
                print(f"Error: {e}. Retrying in {retry_delay} seconds...")
                attempts += 1
                time.sleep(retry_delay)
                self.reconnect()  # Attempt to reconnect

        raise Exception(f"Failed to send message after {max_retries} attempts.")
