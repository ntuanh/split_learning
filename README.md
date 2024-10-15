# Split Learning

## Mô hình triển khai

![sl_model](pics/sl_model.png)

## Yêu cầu các gói
```
torch
torchvision
pika
tqdm
pyyaml
```

Dựng một RabbitMQ server để giao tiếp các bản tin qua môi trường mạng. File `docker-compose.yaml`:

```yaml
version: '3'

services:
  rabbitmq:
    image: rabbitmq:management
    container_name: rabbitmq
    ports:
      - "5672:5672"   # RabbitMQ main port
      - "15672:15672" # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
volumes:
  rabbitmq_data:
    driver: local
```

## Cấu hình

Cấu hình ứng dụng chạy trong file `config.yaml`:

```yaml
name: Split Learning
server:   # cấu hình trên server
  num-round: 1  # số round thực hiện
  clients:  # Layer 1 có 3 client, layer 2 có 2 client, layer 3 có 1 client
    - 3
    - 2
    - 1
  filename: resnet_model  # tên file *.pth được lưu
  validation: True  # cho phép server thực hiện test 

rabbit:   # cấu hình kết nối rabbitMQ
  address: 127.0.0.1    # địa chỉ
  username: admin
  password: admin

learning:
  learning-rate: 0.01
  momentum: 1
  batch-size: 256
  control-count: 3    # control count trên client
```

## Cách chạy

### Server
```commandline
python server.py
```

### Client

Lưu ý: Trong cùng một layer, các client được phép khai trùng ID của nhau.

#### Layer 1
```commandline
python client_layers_1.py --id 1
```

```commandline
python client_layers_1.py --id 2
```

```commandline
python client_layers_1.py --id 3
```

#### Layer 2
```commandline
python client_layers_2.py --id 1
```
```commandline
python client_layers_2.py --id 2
```

## File parameters

Trên server, các file `*.pth` được lưu trong đường dẫn chạy chính của code `server.py` sau khi training xong 1 round.

Nếu tồn tại file `*.pth`, server sẽ đọc file và truyền parameters tới các client. Ngược lại, nếu không tồn tại file `*.pth`, model DNN được tạo với các parameters mới. Vì vậy nếu muốn clear quá trình training trước, cần xóa các file `*.pth`

Ứng dụng đang trong quá trình phát triển ...
