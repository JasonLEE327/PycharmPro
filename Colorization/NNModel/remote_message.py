import socket

def send(message):
    message = str(message)
    print(message)
    address = ('127.0.0.1', 8846)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(message.encode(), address)
    s.close()
