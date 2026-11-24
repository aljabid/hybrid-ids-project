import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import get_logger
logger = get_logger(__name__)
PCAP_DIR     = "data/raw/local"
OUTPUT_PATH  = "data/processed/local/flows_raw.csv"
TIME_WINDOW  = 60
MIN_PACKETS  = 2
PCAP_FILES = {
    "normal_traffic.pcap": "Normal",
    "syn_scan.pcap":        "SYN Scan",
    "tcp_scan.pcap":        "TCP Connect",
    "xmas_scan.pcap":       "Xmas Scan",
}
def make_flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    ep1 = (src_ip, src_port)
    ep2 = (dst_ip, dst_port)
    if ep1 > ep2:
        ep1, ep2 = ep2, ep1
    return (*ep1, *ep2, proto)
def extract_flows_from_pcap(pcap_path, label):
    try:
        from scapy.all import rdpcap, IP, TCP, UDP, ICMP
        logger.info(f"Parsing {pcap_path}")
        packets = rdpcap(pcap_path)
    except ImportError:
        logger.warning("Scapy not installed — generating synthetic demo flows")
        return _synthetic_flows(label, n=200)
    except Exception as e:
        logger.warning(f"Could not read {pcap_path}: {e} — using synthetic data")
        return _synthetic_flows(label, n=200)
    flows = defaultdict(list)
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        ip  = pkt[IP]
        proto_num = ip.proto
        src_port = dst_port = 0
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        key = make_flow_key(ip.src, ip.dst, src_port, dst_port, proto_num)
        flows[key].append(pkt)
    records = []
    for key, pkts in flows.items():
        if len(pkts) < MIN_PACKETS:
            continue
        record = _compute_flow_features(pkts, key, label)
        if record:
            records.append(record)
    logger.info(f"  Extracted {len(records)} flows from {os.path.basename(pcap_path)}")
    return records
def _compute_flow_features(pkts, key, label):
    from scapy.all import IP, TCP, UDP
    try:
        timestamps  = [float(p.time) for p in pkts]
        sizes       = [len(p) for p in pkts]
        duration    = max(timestamps) - min(timestamps)
        pkt_count   = len(pkts)
        byte_count  = sum(sizes)
        pkt_rate    = pkt_count / duration if duration > 0 else pkt_count
        iats = np.diff(sorted(timestamps))
        inter_arrival_mean = float(np.mean(iats)) if len(iats) > 0 else 0.0
        syn_count = ack_count = fin_count = 0
        dst_ports = set()
        proto_type = 0
        for p in pkts:
            if p.haslayer(TCP):
                flags = p[TCP].flags
                if flags & 0x02:  syn_count += 1
                if flags & 0x10:  ack_count += 1
                if flags & 0x01:  fin_count += 1
                dst_ports.add(p[TCP].dport)
            elif p.haslayer(UDP):
                proto_type = 1
                dst_ports.add(p[UDP].dport)
            else:
                proto_type = 2
        return {
            "flow_duration":       round(duration, 6),
            "packet_count":        pkt_count,
            "byte_count":          byte_count,
            "packet_rate":         round(pkt_rate, 4),
            "inter_arrival_mean":  round(inter_arrival_mean, 6),
            "syn_flag_count":      syn_count,
            "ack_flag_count":      ack_count,
            "fin_flag_count":      fin_count,
            "dst_port_diversity":  len(dst_ports),
            "protocol_type":       proto_type,
            "label":               label
        }
    except Exception as e:
        logger.debug(f"Flow feature error: {e}")
        return None
def _synthetic_flows(label, n=200):
    np.random.seed({"Normal": 0, "SYN Scan": 1,
                    "TCP Connect": 2, "Xmas Scan": 3}.get(label, 99))
    records = []
    for _ in range(n):
        if label == "Normal":
            r = {
                "flow_duration":      np.random.exponential(5.0),
                "packet_count":       int(np.random.poisson(20)),
                "byte_count":         int(np.random.poisson(5000)),
                "packet_rate":        np.random.uniform(1, 20),
                "inter_arrival_mean": np.random.exponential(0.5),
                "syn_flag_count":     int(np.random.poisson(1)),
                "ack_flag_count":     int(np.random.poisson(8)),
                "fin_flag_count":     int(np.random.poisson(1)),
                "dst_port_diversity": int(np.random.randint(1, 4)),
                "protocol_type":      0,
            }
        elif label == "SYN Scan":
            r = {
                "flow_duration":      np.random.exponential(0.05),
                "packet_count":       int(np.random.randint(4, 8)),
                "byte_count":         int(np.random.randint(40, 80)),
                "packet_rate":        np.random.uniform(50, 500),
                "inter_arrival_mean": np.random.exponential(0.01),
                "syn_flag_count":     1,
                "ack_flag_count":     0,
                "fin_flag_count":     0,
                "dst_port_diversity": int(np.random.randint(1, 3)),
                "protocol_type":      0,
            }
        elif label == "TCP Connect":
            r = {
                "flow_duration":      np.random.exponential(0.2),
                "packet_count":       int(np.random.randint(3, 8)),
                "byte_count":         int(np.random.randint(120, 400)),
                "packet_rate":        np.random.uniform(20, 200),
                "inter_arrival_mean": np.random.exponential(0.05),
                "syn_flag_count":     1,
                "ack_flag_count":     int(np.random.randint(1, 4)),
                "fin_flag_count":     1,
                "dst_port_diversity": int(np.random.randint(1, 3)),
                "protocol_type":      0,
            }
        else:
            r = {
                "flow_duration":      np.random.exponential(0.05),
                "packet_count":       int(np.random.randint(4, 8)),
                "byte_count":         int(np.random.randint(40, 80)),
                "packet_rate":        np.random.uniform(50, 300),
                "inter_arrival_mean": np.random.exponential(0.01),
                "syn_flag_count":     0,
                "ack_flag_count":     0,
                "fin_flag_count":     1,
                "dst_port_diversity": int(np.random.randint(1, 3)),
                "protocol_type":      0,
            }
        r["label"] = label
        records.append(r)
    return records
def run():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_records = []
    for filename, label in PCAP_FILES.items():
        pcap_path = os.path.join(PCAP_DIR, filename)
        records   = extract_flows_from_pcap(pcap_path, label)
        all_records.extend(records)
        logger.info(f"  {label}: {len(records)} flows")
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\nTotal flows: {len(df)}")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"Saved → {OUTPUT_PATH}")
    return df
if __name__ == "__main__":
    run()
