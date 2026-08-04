# Nmap Commands — Exact Commands Used for Traffic Generation

## Environment
- Attacker:  Kali Linux (Nmap installed)
- Victim:    Windows 10 / Ubuntu on same LAN segment
- Monitor:   Wireshark on monitoring machine (promiscuous mode)

---

## Attack Traffic Generation

### SYN Scan
```bash
nmap -sS <target_ip> -p 1-1000 --max-rate 100 -oN syn_scan_results.txt
```
- Sends SYN packets only (half-open scan)
- Does NOT complete TCP handshake
- Generates high SYN flag count, low ACK/FIN count

### TCP Connect Scan
```bash
nmap -sT <target_ip> -p 1-1000 --max-rate 100 -oN tcp_scan_results.txt
```
- Completes full TCP three-way handshake
- Generates SYN + ACK + FIN flag patterns
- Slower than SYN scan — easier to detect

### FIN / Xmas Scan
```bash
nmap -sX <target_ip> -p 1-1000 --max-rate 100 -oN xmas_scan_results.txt
```
- Sends packets with FIN + PSH + URG flags set
- Stealth technique — no SYN flags
- Generates FIN flag count, zero SYN/ACK

---

## Normal Traffic Generation

Normal traffic generated manually during capture sessions:

```bash
# Web browsing simulation
curl -s http://<target_ip>/ > /dev/null

# File download
wget http://<target_ip>/testfile.bin -O /dev/null

# ICMP ping flood (light)
ping <target_ip> -c 100 -i 0.2

# DNS queries
for i in {1..50}; do nslookup google.com; sleep 0.5; done
```

---

## Wireshark Capture

```bash
# Start capture on monitoring machine
tshark -i eth0 -w normal_traffic.pcap

# Apply display filter for TCP only (optional)
# Filter: tcp
```

---

## Capture Durations
| Traffic Type    | Duration  | Approx Flows |
|-----------------|-----------|--------------|
| Normal          | 10 min    | ~500+        |
| SYN Scan        | 5 min     | ~200+        |
| TCP Connect     | 5 min     | ~200+        |
| Xmas Scan       | 5 min     | ~200+        |

---

## Important Notes
1. Run each scan type separately with Wireshark capturing
2. Save separate .pcap files per traffic type for clean labeling
3. Record exact start/end timestamps for each session
4. Do NOT mix traffic types in a single capture session
