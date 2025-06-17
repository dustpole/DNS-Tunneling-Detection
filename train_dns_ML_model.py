# DNS Tunneling detection and training script.

# Import needed modules
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import log2
import matplotlib.pyplot as plt
from collections import Counter
from scapy.all import rdpcap
import os

# Function to Calculate Shannon entropy for a string.
def shannon_entropy(text):
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((count / length) * log2(count / length) for count in counts.values())

# Function to parse PCAP and extract DNS-specific features with labels.
def extract_dns_features(pcap_file):
    packets = rdpcap(pcap_file)
    features = []
    labels = []
    query_lengths = {'benign': [], 'malicious': []}
    prev_time = 0

    dns_packets = []
    for pkt in packets:
        if pkt.haslayer('IP') and pkt.haslayer('DNS'):
            src_ip = pkt['IP'].src # Source IP
            dst_ip = pkt['IP'].dst # Destination IP
            timestamp = pkt.time # Time of packet
            qtype = pkt['DNS'].qd.qtype if pkt['DNS'].qd else 0 # DNS query type
            dns_packets.append({
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'timestamp': timestamp,
                'qtype': qtype
            })

    # Features Section of code
    with open('dns_labels.txt', 'w', encoding='utf-8') as f: # UTF-8 used to allow non-translatable characters in subdomains to be wrote to file
        for pkt in packets:
            if pkt.haslayer('IP') and pkt.haslayer('DNS'): # Only check DNS packets
                pkt_len = len(pkt) # Packet Data-length
                pkt_time = pkt.time # Packet time
                query_len = len(pkt['DNS'].qd.qname) if pkt['DNS'].qd else 0 #Length of the DNS query
                is_response = 1 if pkt['DNS'].qr == 1 else 0 # If DNS query is a response
                record_count = pkt['DNS'].ancount if is_response else 0 # Number of returned requested records
                qtype = pkt['DNS'].qd.qtype if pkt['DNS'].qd else 0 # DNS query type

                # Handling for non-unicode subdomain names
                try:
                    query_name = pkt['DNS'].qd.qname.decode('utf-8', errors='replace').rstrip('.') if pkt['DNS'].qd else ''
                except Exception:
                    query_name = ''
                subdomain = query_name.split('.')[0] if query_name else '' # Parsing for unicode left-most subdomains

                if subdomain == '':
                    entropy = 3.1 # For non-unicode subdomains set entropy to 3.1
                else:
                    entropy = shannon_entropy(subdomain) # Checking the entropy of the subdomain

                feature = [pkt_len, inter_arrival, query_len, is_response, record_count, qtype, entropy]
                features.append(feature)

                # Rule-based labeling heuristic
                score = 0
                if entropy > 3.1:
                    score += .75
                if query_len > 40:
                    score += .5
                if qtype in [16, 28]:
                    score += .25
                if subdomain == '':
                    score += .75

                # Setting the malicious(1) or benign(0) label
                label = 1 if score >= 1 else 0
                if label == 1:
                    query_lengths['malicious'].append(query_len)
                else:
                    query_lengths['benign'].append(query_len)
                labels.append(label)

                # Debugging: Log packet details to console
                debug_info = f"Packet: query={query_name}, subdomain={subdomain}, entropy={entropy:.2f}, qtype={qtype}, label={label}"
                f.write(debug_info + '\n')

    feature_columns = [
        'pkt_len', 'inter_arrival', 'query_len', 'is_response',
        'record_count', 'qtype', 'entropy'
    ]
    # Compile the features and labels arrays into a csv for machine learning input.
    df = pd.DataFrame(features, columns=feature_columns)
    df['label'] = labels
    df.to_csv('dns_features.csv', index=False)

    print("Label distribution:", Counter(labels))
    return np.array(features), np.array(labels), query_lengths

# Step 1: Prepare data
pcap_file = 'capture.pcap'  # PCAP File Name to process and train using.
X, y, query_lengths = extract_dns_features(pcap_file)

data = X
labels = y

# Plot query length distribution to a png with matplotlib
plt.figure(figsize=(10, 6))
plt.hist(query_lengths['benign'], bins=30, alpha=0.7, label='Benign', color='blue', rwidth=0.20)
plt.hist(query_lengths['malicious'], bins=30, alpha=0.7, label='Malicious', color='red', rwidth=0.20)
plt.xlabel('Query Name Length')
plt.ylabel('Frequency')
plt.title('Distribution of DNS Query Name Lengths (Benign vs. Malicious)')
plt.legend()
plt.xticks(np.arange(0, 51, 5))
plt.xlim(0, 50)
plt.savefig('query_length_distribution.png')
plt.close()

# Plot Entropy distribution to a png with matplotlib
plt.figure(figsize=(10, 6))
plt.hist([f[6] for f, l in zip(X, y) if l == 0], bins=30, alpha=0.7, label='Benign', color='blue')
plt.hist([f[6] for f, l in zip(X, y) if l == 1], bins=30, alpha=0.7, label='Malicious', color='red')
plt.xlabel('Subdomain Entropy')
plt.ylabel('Frequency')
plt.title('Entropy Distribution')
plt.legend()
plt.xticks(np.arange(0, 5.5, 0.5))
plt.savefig('entropy_distribution.png')
plt.close()

# Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler parameters
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
np.save("scaler_var.npy", scaler.var_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the model
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

model = DNSNet(input_size=7) # Set model features quantity

# Step 3: Load Saved Model if one exists
if os.path.isfile('dns_model.pth'):
    model.load_state_dict(torch.load('dns_model.pth'))
    model.eval()

# Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50 # Epochs quantity
batch_size = 32 # Batch Size quantity
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Display results every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# Evaluate on test data
model.eval()
with torch.no_grad():
    preds = model(X_test).round()
    accuracy = (preds.eq(y_test)).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy:.2%}")

# Save model
torch.save(model.state_dict(), "dns_model.pth")

# Convert tensors to NumPy arrays
y_true = y_test.numpy().flatten()
y_pred = preds.numpy().flatten()

# Print precision, recall, F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Full classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))