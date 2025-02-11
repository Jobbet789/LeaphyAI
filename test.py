"""
This script will test the Godot client for usage with the neural network server.
"""
import json
import socket

def handle_client(conn, addr):
    print("Connected by", addr)

    try:
        while True:
            data = conn.recv(1024).decode("utf-8")
            if not data:
                print("No data received")
                break
            
            print("Received data:", data)
            try:
                message = json.loads(data)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                continue


            state = message.get("state")
            reward = message.get("reward")
            done = message.get("done")
            print(f"State: {state}, Reward: {reward}, Done: {done}")

            action = [1, 1]
            response = json.dumps(action)
            conn.sendall(response.encode("utf-8"))
            print("Sent response:", response)
        
    except Exception as e:
        print("Exception:", e)
    finally:
        conn.close()
        print("Connection closed")

def main():
    host = "127.0.0.1"
    port = 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        print("Server listening on port", port)

        conn, addr = s.accept()
        handle_client(conn, addr)



if __name__ == "__main__":
    while True:
        main()
