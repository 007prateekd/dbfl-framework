import argparse
import hashlib
import json
import logging
import warnings
from collections import OrderedDict
from functools import reduce
from time import time
from urllib.parse import urlparse
from uuid import uuid4
import random

import numpy as np
import pandas as pd
import requests
import torch
from torchmetrics.classification import BinaryRecall, BinaryPrecision
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.classifier = nn.Sequential(
    #         nn.Linear(30, 256),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, 256),
    #         nn.ReLU(),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(0.2),
    #         nn.Linear(256, 1),
    #         nn.Sigmoid()
    #     )    

    # def forward(self, x: torch.Tensor):
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x
    
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
            
    @staticmethod   
    def model_init(net, grads):
        params_dict = zip(net.state_dict().keys(), grads)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        return net

    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SiloTestData(Dataset):
    def __init__(self):
        self.data = pd.read_csv("../dl-app/test_0/data.csv")
        self.labels = pd.read_csv("../dl-app/test_0/labels.csv")
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.data.iloc[idx, 1:], dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx, 1], dtype=torch.float32)
        return data, label
    
    
class Evaluation(object):
    def __init__(self):
        self.testloader = None
    
    # def load_test_data(self):
    #     silo_dataset = SiloTestData()
    #     return DataLoader(silo_dataset)
    def load_test_data(self):
        """Load CIFAR-10 (training and test set)."""
        trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = CIFAR10("./data", train=True, download=True, transform=trf)
        testset = CIFAR10("./data", train=False, download=True, transform=trf)
        return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

    @staticmethod
    def test(net, testloader):
        # """Validate the model on the test set."""
        # criterion = torch.nn.BCELoss()
        # correct, total, loss = 0, 0, 0
        # tp, fp, fn, tn = 0, 0, 0, 0
        # with torch.no_grad():
        #     for data, labels in tqdm(testloader):
        #         outputs = net(data.to(DEVICE))
        #         rounded = torch.round(outputs)
        #         rounded = torch.reshape(rounded, (1, len(rounded)))[0]
        #         labels = labels.to(DEVICE)
        #         loss += criterion(rounded, labels).item()
        #         total += labels.size(0)
        #         correct += (rounded == labels).sum().item()
        #         tp += (rounded[0].item() == labels[0].item() and labels[0].item() == 1)
        #         fp += (rounded[0].item() != labels[0].item() and labels[0].item() == 0)
        #         fn += (rounded[0].item() != labels[0].item() and labels[0].item() == 1)
        #         tn += (rounded[0].item() == labels[0].item() and labels[0].item() == 0)
        # print(tp, fn, fp, tn)
        # score = loss / len(testloader.dataset)
        # accuracy  = correct / total
        # try:
        #     recall = tp / (tp + fn)
        # except:
        #     recall = np.nan
        # try:
        #     precision = tp / (tp + fp)
        # except:
        #     precision = np.nan
        # return score, accuracy, recall, precision   
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                outputs = net(images.to(DEVICE))
                labels = labels.to(DEVICE)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        return loss, accuracy    


class Blockchain(object):
    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.nodes = set()
        net = Net()
        grads = [val.cpu().numpy() for _, val in net.state_dict().items()]
        json_dump = json.dumps(
            obj={"grads": grads}, 
            cls=NumpyEncoder
        )
        self.grads_g = grads
        self.grads_l = []
        self.current_transactions.append(json.loads(json_dump))
        self.trust_scores = []
        self.new_block(previous_hash=1, proof=100)
        
    def new_block(self, proof, previous_hash=None):
        """
        Create a new Block in the Blockchain
        :param proof: <int> The proof given by the Proof of Work algorithm
        :param previous_hash: (Optional) <str> Hash of previous Block
        :return: <dict> New Block
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        return block
    
    def new_transaction(self, grads):
        """
        Creates a new transaction to go into the next mined Block
        :param sender: <str> Address of the Sender
        :param recipient: <str> Address of the Recipient
        :param amount: <int> Amount
        :return: <int> The index of the Block that will hold this transaction
        """
        self.current_transactions.append({
            'grads': grads,
            # 'recipient': recipient,
            # 'amount': amount,
        })
        return self.last_block['index'] + 1
    
    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a Block
        :param block: <dict> Block
        :return: <str>
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def proof_of_work(self, last_proof):
        """
        Simple Proof of Work Algorithm:
         - Find a number p' such that hash(pp') contains leading 4 zeroes, where p is the previous p'
         - p is the previous proof, and p' is the new proof
        :param last_proof: <int>
        :return: <int>
        """
        proof = 0
        while self.valid_proof(last_proof, proof, 4) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof, diff):
        """
        Validates the Proof: Does hash(last_proof, proof) contain `diff` leading zeroes?
        :param last_proof: <int> Previous Proof
        :param proof: <int> Current Proof
        :param diff: <int> Difficulty of PoW
        :return: <bool> True if correct, False if not.
        """
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:diff] == "0" * diff
    
    def register_node(self, address):
        """
        Add a new node to the list of nodes
        :param address: <str> Address of node. Eg. 'http://192.168.0.5:5000'
        :return: None
        """
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)    

    def valid_chain(self, chain):
        """
        Determine if a given blockchain is valid
        :param chain: <list> A blockchain
        :return: <bool> True if valid, False if not
        """
        last_block = chain[0]
        current_index = 1
        while current_index < len(chain):
            block = chain[current_index]
            if block['previous_hash'] != self.hash(last_block):
                return False
            if not self.valid_proof(last_block['proof'], block['proof'], 4):
                return False
            last_block = block
            current_index += 1
        return True

    def resolve_conflicts(self):
        """
        This is our Consensus Algorithm, it resolves conflicts
        by replacing our chain with the longest one in the network.
        :return: <bool> True if our chain was replaced, False if not
        """
        neighbours = self.nodes
        new_chain = None
        max_length = len(self.chain)
        for node in neighbours:
            response = requests.get(f'http://{node}/chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
        if new_chain:
            self.chain = new_chain
            return True
        return False
   
    @staticmethod
    def aggregate_weights(results):
        """Compute weighted average."""
        num_examples_total = sum([num_examples for _, num_examples in results])
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]
        weights_prime = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
    
    @staticmethod
    def aggregate_weights_custom(results, trust_scores):
        """Compute weighted average."""
        denominator = sum([
            num_examples * trust_score \
            for (_, num_examples), trust_score in zip(results, trust_scores)
        ])
        # for layer, _ in results:
        #     print(len(layer))
        #     for l in layer:
        #         print(l.shape)
        weighted_weights = [
            [layer * num_examples * trust_score for layer in weights] \
            for (weights, num_examples), trust_score in zip(results, trust_scores)
        ]
        weights_prime = [
            reduce(np.add, layer_updates) / denominator
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
  
  
app = Flask(__name__)
node_identifier = str(uuid4()).replace('-', '')
blockchain = Blockchain()
evaluation = Evaluation()
_, testloader = evaluation.load_test_data()


@app.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)
    ## Reward miner
    # blockchain.new_transaction(
    #     sender="0",
    #     recipient=node_identifier,
    #     amount=1,
    # )
    ## Mine the block
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof, previous_hash)
    response = {
        'message': "New Block Mined",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200

@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['grads_agg']
    if not all(k in values for k in required):
        return "Missing values", 400
    index = blockchain.new_transaction(values['grads_agg'])
    response = {'message': f"Transaction will be added to Block {index}"}
    return jsonify(response), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

@app.route('/nodes/all', methods=['GET'])
def nodes_all():
    response = {'nodes': list(blockchain.nodes)}
    return jsonify(response), 200

@app.route('/nodes/id', methods=['GET'])
def node_identifiers():
    response = {'node_id': node_identifier}
    return jsonify(response), 200

@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400
    for node in nodes:
        blockchain.register_node(node)
    response = {
        'message': "New nodes have been added",
        'total_nodes': list(blockchain.nodes),
    }
    return jsonify(response), 201

@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()
    if replaced:
        response = {
            'message': "Chain replaced",
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': "Chain authoritative",
            'chain': blockchain.chain
        }
    return jsonify(response), 200

@app.route('/trust_scores/init', methods=['POST'])
def trust_scores_init():
    values = request.get_json()
    num_clients = values.get('num_clients')
    b_nodes = nodes_all()[0].get_json()['nodes']
    for b_node in b_nodes:
        json_dict = {"trust_scores": [1 / num_clients] * num_clients}
        r = requests.post(url=f"http://{b_node}/trust_scores/broadcast", json=json_dict)
    return jsonify({}), 201

@app.route('/trust_scores/broadcast', methods=['POST'])
def trust_scores_broadcast():
    values = request.get_json()
    blockchain.trust_scores = values.get('trust_scores')
    return jsonify({}), 201

@app.route('/trust_scores/get', methods=['GET'])
def trust_scores_get():
    response = {'trust_scores': blockchain.trust_scores}
    return jsonify(response), 200

@app.route('/trust_scores/update', methods=['POST'])
def trust_scores_update():
    values = request.get_json()
    blockchain.trust_scores = values.get('trust_scores')
    return jsonify({}), 201
    
@app.route('/grads/validate', methods=['POST'])
def validate():
    values = request.get_json()
    grads_l, n_data, c_id, masks = values.get('grads_l'), \
        values.get('n_data'), values.get('c_id'), values.get('masks')
    grads_g_np = blockchain.grads_g
    grads_l_np = [np.array(grad) for grad in grads_l[0]]
    # grads_agg = blockchain.aggregate_weights([(grads_l_np, grads_l[1]), (grads_g_np, n_data)])
    grads_agg = blockchain.aggregate_weights_custom(
        results=[(grads_l_np, grads_l[1]), (grads_g_np, n_data)], 
        trust_scores=[blockchain.trust_scores[c_id], 1]
    )
    grads_agg_np = [np.array(grad) for grad in grads_agg]
    # Initialize Global Model
    net_g = Net()
    net_g.eval()
    net_g = Net.model_init(net_g, grads_g_np)
    # Initialize Aggregate Model
    net_agg = Net()
    net_agg.eval()
    net_agg = Net.model_init(net_agg, grads_agg_np)
    # Evaluate
    # loss_g, score_g, recall, precision = evaluation.test(net_g, testloader)
    # loss_agg, score_agg, recall, precision = evaluation.test(net_agg, testloader)
    loss_g, score_g = evaluation.test(net_g, testloader)
    loss_agg, score_agg = evaluation.test(net_agg, testloader)
    diff_loss, diff_score = loss_g - loss_agg, score_agg - score_g
    """
    RepSecAgg: Add masks to local grads
    """
    sum_masked = sum(masks[:c_id]) + sum(masks[c_id + 1:])
    scale_factor = grads_l[1] * blockchain.trust_scores[c_id]
    grads_masked = [[
        grad + sum_masked / scale_factor for grad in grads
        ] for grads in grads_l_np
    ]
    blockchain.grads_l.append((grads_masked, grads_l[1], diff_loss, diff_score))
    print(f"🔺  Score:\n  * Agg: {score_agg}\n  * Glo: {score_g}")
    print(f"🔻  Loss\n  * Agg: {round(loss_agg, 4)}\n  * Glo: {round(loss_g, 4)}")
    # print(f"🔺 Recall: {precision}")
    # print(f"🔺 Precision: {recall}")
    return jsonify({}), 201

@app.route('/grads/update', methods=['GET'])
def grads_update():
    data = requests.get(url=f"http://{request.host}/chain").json()
    grads_g = data['chain'][-1]['transactions'][0]['grads']
    grads_g = [np.array(grads) for grads in grads_g]
    blockchain.grads_g = grads_g
    return jsonify({}), 200
    
@app.route('/grads/get', methods=['GET'])
def grads_get():
    json_dump = json.dumps(
        obj={'grads_l': blockchain.grads_l},
        cls=NumpyEncoder
    )
    blockchain.grads_l = []
    response = json.loads(json_dump)
    return jsonify(response), 200

@app.route('/aggregate', methods=['GET'])
def aggregate():
    grads_l_all = []
    diff_all = []
    b_nodes = nodes_all()[0].get_json()['nodes']
    for b_node in b_nodes:
        data = requests.get(url=f"http://{b_node}/grads/get").json()
        for grad_l, cnt, diff_loss, diff_score in data['grads_l']:
            grads_l_np = [np.array(grad) for grad in grad_l]
            grads_l_all.append((grads_l_np, cnt))
            diff_all.append(diff_loss)
        
    # Update trust scores
    diff_sum = sum([abs(diff) for diff in diff_all])
    for i in range(len(blockchain.trust_scores)):
        blockchain.trust_scores[i] = blockchain.trust_scores[i] * (1 + diff_all[i] / diff_sum)
    # Normalize i.e. make total sum = 1
    trust_scores_sum = sum(blockchain.trust_scores)
    for i in range(len(blockchain.trust_scores)):
        blockchain.trust_scores[i] = blockchain.trust_scores[i] / trust_scores_sum
    # Update weights
    weights_results = []
    for grads in grads_l_all:
        weights_results.append((grads[0], grads[1]))
    # grads_agg = blockchain.aggregate_weights(weights_results)
    grads_agg = blockchain.aggregate_weights_custom(weights_results, blockchain.trust_scores)
    json_dump = json.dumps(
        obj={"grads_agg": grads_agg}, 
        cls=NumpyEncoder
    )
    json_dict = json.loads(json_dump)
    r = requests.post(url=f"http://{request.host}/transactions/new", json=json_dict)
    r = requests.get(url=f"http://{request.host}/mine")
    for b_node in b_nodes:
        r = requests.get(url=f"http://{b_node}/nodes/resolve")
        r = requests.get(url=f"http://{b_node}/grads/update")
        json_dump_tmp = json.dumps(
            obj={"trust_scores": blockchain.trust_scores}, 
        )
        json_dict_tmp = json.loads(json_dump_tmp)
        r = requests.post(url=f"http://{b_node}/trust_scores/update", json=json_dict_tmp)
    return jsonify(json_dict), 200
    
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5000)
    args = parser.parse_args()
    host = '127.0.0.1'
    port = args.port
    app.run(host=host, port=port)