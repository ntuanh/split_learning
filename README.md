# Split Learning

## Mô hình triển khai

```
n-devices --- n-devices --- n-devices
```

## Yêu cầu các gói
```
torch
torchvision
pika
tqdm
```

Và có rabbitMQ server để kết nối tới

## Cách chạy

### Server
```commandline
python server.py --topo 1 2
```
Cụ thể, layer 1 có 1 client, layer 2 có 2 client

### Client

#### Layer 1
```commandline
python client_layers_1.py --id 1
```

#### Layer 2
```commandline
python client_layers_2.py --id 1
```
```commandline
python client_layers_2.py --id 2
```

## File parameters

Trên server, các file `*.pth` được lưu trong đường dẫn chạy chính của code `server.py`

Quá trình đang phát triển sản phẩm ...
