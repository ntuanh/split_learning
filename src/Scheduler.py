import time
import uuid
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import src.Log


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device, num_layers):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.num_devices = num_layers

    def send_intermediate_output(self, data_id, output, labels, trace, test=False):
        forward_queue_name = f'intermediate_queue_{self.layer_id}'
        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": trace,
                 "test": test}
            )
        else:
            message = pickle.dumps(
                {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": [self.client_id],
                 "test": test}
            )

        self.channel.basic_publish(
            exchange='',
            routing_key=forward_queue_name,
            body=message
        )

    def send_gradient(self, data_id, gradient, trace):
        to_client_id = trace[-1]
        trace.pop(-1)
        backward_queue_name = f'gradient_queue_{self.layer_id - 1}_{to_client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)

        message = pickle.dumps(
            {"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace, "test": False})

        self.channel.basic_publish(
            exchange='',
            routing_key=backward_queue_name,
            body=message
        )

    def send_validation(self, data_id, data, trace):
        to_client_id = trace[0]
        backward_queue_name = f'gradient_queue_1_{to_client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)

        message = pickle.dumps({"data_id": data_id, "data": data, "trace": trace, "test": True})

        self.channel.basic_publish(
            exchange='',
            routing_key=backward_queue_name,
            body=message
        )

    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

    def train_on_first_layer(self, model, control_count, batch_size, lr, momentum, validation):
        # Read and load dataset
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True)

        if validation:
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False)
        else:
            test_loader = None

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        data_iter = iter(train_loader)

        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        num_forward = 0
        num_backward = 0
        end_data = False
        data_store = {}

        model.to(self.device)
        with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
            while True:
                # Training model
                model.train()
                optimizer.zero_grad()
                # Process gradient
                method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                if method_frame and body:
                    num_backward += 1
                    received_data = pickle.loads(body)
                    gradient_numpy = received_data["data"]
                    gradient = torch.tensor(gradient_numpy).to(self.device)
                    data_id = received_data["data_id"]

                    data_input = data_store.pop(data_id)
                    output = model(data_input)
                    output.backward(gradient=gradient, retain_graph=True)
                    optimizer.step()
                else:
                    # speed control
                    if len(data_store) > control_count:
                        continue
                    # Process forward message
                    try:
                        training_data, labels = next(data_iter)
                        training_data = training_data.to(self.device)
                        data_id = uuid.uuid4()
                        data_store[data_id] = training_data
                        intermediate_output = model(training_data)
                        intermediate_output = intermediate_output.detach().requires_grad_(True)

                        # Send to next layers
                        num_forward += 1
                        # tqdm bar
                        pbar.update(1)

                        self.send_intermediate_output(data_id, intermediate_output, labels, None)

                    except StopIteration:
                        end_data = True
                if end_data and (num_forward == num_backward):
                    break

            # validation
            num_forward = 0
            num_backward = 0
            all_labels = np.array([])
            all_vals = np.array([])
            if test_loader:
                for (testing_data, labels) in test_loader:
                    testing_data = testing_data.to(self.device)
                    data_id = uuid.uuid4()
                    intermediate_output = model(testing_data)

                    # Send to next layers
                    num_forward += 1

                    self.send_intermediate_output(data_id, intermediate_output, labels, None, True)

                while True:
                    method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                    if method_frame and body:
                        num_backward += 1
                        received_data = pickle.loads(body)
                        test_data = received_data["data"]

                        all_labels = np.append(all_labels, test_data[0])
                        all_vals = np.append(all_vals, test_data[1])

                    if num_forward == num_backward:
                        break

                notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                               "message": "Finish training!", "validate": [all_labels, all_vals]}
            else:
                notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                               "message": "Finish training!", "validate": None}

        # Finish epoch training, send notify to server
        src.Log.print_with_color("[>>>] Finish training!", "red")
        self.send_to_server(notify_data)

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    break
            time.sleep(0.5)

    def train_on_last_layer(self, model, lr, momentum):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        criterion = nn.CrossEntropyLoss()
        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        while True:
            # Training model
            model.train()
            optimizer.zero_grad()
            # Process gradient
            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]
                data_id = received_data["data_id"]
                labels = received_data["label"].to(self.device)
                test = received_data["test"]

                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)

                if test:
                    output = model(intermediate_output)
                    labels = labels.cpu().numpy().flatten()
                    pred = output.data.max(1, keepdim=True)[1]
                    pred = pred.cpu().numpy().flatten()
                    self.send_validation(data_id, [labels, pred], trace)
                else:
                    output = model(intermediate_output)
                    loss = criterion(output, labels)
                    print(f"Loss: {loss.item()}")
                    intermediate_output.retain_grad()
                    loss.backward()
                    optimizer.step()

                    gradient = intermediate_output.grad
                    self.send_gradient(data_id, gradient, trace)  # 1F1B
            # Check training process
            else:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "PAUSE":
                        break

    def train_on_middle_layer(self, model, control_count, lr, momentum):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        data_store = {}
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        while True:
            # Training model
            model.train()
            optimizer.zero_grad()
            # Process gradient
            method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                gradient_numpy = received_data["data"]
                gradient = torch.tensor(gradient_numpy).to(self.device)
                trace = received_data["trace"]
                data_id = received_data["data_id"]

                data_input = data_store.pop(data_id)
                output = model(data_input)
                data_input.retain_grad()
                output.backward(gradient=gradient, retain_graph=True)
                optimizer.step()

                gradient = data_input.grad
                self.send_gradient(data_id, gradient, trace)
            else:
                method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
                if method_frame and body:
                    received_data = pickle.loads(body)
                    intermediate_output_numpy = received_data["data"]
                    trace = received_data["trace"]
                    data_id = received_data["data_id"]
                    test = received_data["test"]
                    labels = received_data["label"].to(self.device)

                    intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)
                    data_store[data_id] = intermediate_output

                    output = model(intermediate_output)
                    output = output.detach().requires_grad_(True)

                    self.send_intermediate_output(data_id, output, labels, trace, test)
                    # speed control
                    if len(data_store) > control_count:
                        continue
            # Check training process
            if method_frame is None:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "PAUSE":
                        break

    def train_on_device(self, model, control_count, batch_size, lr, momentum, validation):
        if self.layer_id == 1:
            self.train_on_first_layer(model, control_count, batch_size, lr, momentum, validation)
        elif self.layer_id == self.num_devices:
            self.train_on_last_layer(model, lr, momentum)
        else:
            self.train_on_middle_layer(model, control_count, lr, momentum)
