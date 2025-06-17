import numpy as np
from scapy.all import sniff
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import log2
from collections import Counter


class DNSNet(nn.Module):
    def __init__(self, input_size):
        super(DNSNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def shannon_entropy(text):
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((count / length) * log2(count / length) for count in counts.values())

def extract_dns_features_from_packet(pkt):
    features = []
    labels = []
    query_lengths = {'benign': [], 'malicious': []}
    prev_time = 0

    if pkt.haslayer('IP') and pkt.haslayer('DNS'):
        # Extract features
        pkt_len = len(pkt)  # Packet size
        pkt_time = pkt.time  # Timestamp
        inter_arrival = pkt_time - prev_time if prev_time != 0 else 0
        prev_time = pkt_time

        # DNS-specific features
        query_len = len(pkt['DNS'].qd.qname) if pkt['DNS'].qd else 0  # Query name length
        is_response = 1 if pkt['DNS'].qr == 1 else 0  # 0: query, 1: response
        record_count = pkt['DNS'].ancount if is_response else 0  # Number of answer records
        qtype = pkt['DNS'].qd.qtype if pkt['DNS'].qd else 0  # Query type (e.g., A, AAAA)

        # Feature: Subdomain entropy
        query_name = pkt['DNS'].qd.qname.decode().rstrip('.') if pkt['DNS'].qd else ''
        subdomain = ''
        if query_name:
            parts = query_name.split('.')
            if parts:
                subdomain = parts[0]  # Take leftmost subdomain
        entropy = shannon_entropy(subdomain)

        # Combine features (8 total)
        feature = [pkt_len, inter_arrival, query_len, is_response, record_count, qtype, entropy]
        features.append(feature)

        # Rule-based labeling
        # Malicious: high entropy (>3), query length > 50, or high DNS traffic (>10 in 6s with >20% TXT/MX)
        if (entropy > 3):
            label = 1  # Malicious
            query_lengths['malicious'].append(query_len)
        else:
            label = 0  # Benign
            query_lengths['benign'].append(query_len)
        labels.append(label)

        # Debugging: Log packet details
        debug_info = f"Packet: query={query_name}, subdomain={subdomain}, entropy={entropy:.2f}, qtype={qtype}, label={label}"
        print(debug_info)
        f.write(debug_info + '\n')

    return np.array(features), np.array(labels), query_lengths

def process_packet(pkt):
    # Standardize features
    scaler = StandardScaler()
    features, _, _ = extract_dns_features_from_packet(pkt)  # Modify extract_dns_features
    X = torch.tensor(scaler.transform(features), dtype=torch.float32)
    pred = model(X).item()
    if pred >= 0.5:
        print(f"Malicious packet detected: {pkt['DNS'].qd.qname.decode()}")

model = DNSNet(input_size=8)
model.load_state_dict(torch.load('dns_model.pth'))
model.eval()


sniff(filter="udp port 53", prn=process_packet)