import argparse
import requests


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--ports', nargs='+', required=True, type=int)
args = parser.parse_args()
ports = args.ports
n_ports = len(ports)
print("\n⏳  Registering Nodes")
for i in range(n_ports):
    for j in range(n_ports):
        json = {
            "nodes": [f"http://127.0.0.1:{ports[j]}"]
        }
        r = requests.post(
            url=f"http://127.0.0.1:{ports[i]}/nodes/register", 
            json=json
        )
    print(f"👍  Node :{ports[i]} Registered")

print(f"😄  All Nodes Registered\n")