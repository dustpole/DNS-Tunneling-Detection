# Realtime monitoring script to check if DNS traffic is malicious using a trained machine learning model.

# Import needed modules
import numpy as np
from scapy.all import sniff
from scapy.layers.dns import DNS
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from math import log2

# Define DNSNet Model
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

# Entropy calculation
def shannon_entropy(text):
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((count / length) * log2(count / length) for count in counts.values())

# Extract 7 features from DNS packets
def extract_dns_features_from_packet(pkt):
    if pkt.haslayer('IP') and pkt.haslayer(DNS) and pkt[DNS].qd:
        pkt_len = len(pkt)
        inter_arrival = 0  # Placeholder for real inter-arrival logic
        query_len = len(pkt[DNS].qd.qname)
        is_response = int(pkt[DNS].qr == 1)
        record_count = pkt[DNS].ancount if is_response else 0
        qtype = pkt[DNS].qd.qtype

        query_name = pkt[DNS].qd.qname.decode().rstrip('.')
        subdomain = query_name.split('.')[0] if query_name else ''
        entropy = shannon_entropy(subdomain)

        features = [pkt_len, inter_arrival, query_len, is_response, record_count, qtype, entropy]

        label = 1 if entropy > 3 else 0

        # Debug print
        debug_info = f"Packet: query={query_name}, subdomain={subdomain}, entropy={entropy:.2f}, qtype={qtype}, label={label}"
        print(debug_info)

    return np.array([features])

# Load scaler from training phase
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy")
scaler.scale_ = np.load("scaler_scale.npy")
scaler.var_ = np.load("scaler_var.npy")
scaler.n_features_in_ = 7

# Load model
model = DNSNet(input_size=7)
model.load_state_dict(torch.load('dns_model.pth'))
model.eval()

# Process packets with the loaded model
def process_packet(pkt):
    features = extract_dns_features_from_packet(pkt)
    if features is not None:
        X = torch.tensor(scaler.transform(features), dtype=torch.float32)
        with torch.no_grad():
            pred = model(X).item()
        if pred >= 0.7: #Classify the packet as malicious and print a warning into the console.
            print(f"Malicious packet detected: {pkt[DNS].qd.qname.decode()}")

# Start sniffing udp port 53 traffic and process the packets.
sniff(filter="udp port 53", prn=process_packet, store=False)
