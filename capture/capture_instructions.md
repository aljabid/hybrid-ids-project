# Capture Instructions — Environment Setup and Procedure

## Network Environment

```
┌─────────────────┐       LAN        ┌─────────────────┐
│  Attacker       │◄────────────────►│  Victim         │
│  Kali Linux     │                  │  Windows 10 /   │
│  Nmap           │                  │  Ubuntu 22.04   │
│  192.168.1.10   │                  │  192.168.1.20   │
└─────────────────┘                  └─────────────────┘
         │                                    │
         └────────────────┬───────────────────┘
                          │
                 ┌─────────────────┐
                 │  Monitor        │
                 │  Wireshark      │
                 │  192.168.1.30   │
                 │  (promiscuous)  │
                 └─────────────────┘
```

---

## Machine Roles

| Machine   | OS            | Role                          | IP           |
|-----------|---------------|-------------------------------|--------------|
| Attacker  | Kali Linux    | Runs Nmap scans               | 192.168.1.10 |
| Victim    | Windows 10    | Target of scans               | 192.168.1.20 |
| Monitor   | Ubuntu/Kali   | Passive Wireshark capture     | 192.168.1.30 |

> Note: Monitor can be the same machine as victim in promiscuous mode
> if only two machines are available.

---

## Wireshark Setup

### Enable Promiscuous Mode
```bash
sudo ip link set eth0 promisc on
```

### Wireshark Capture Filter (capture all LAN traffic)
```
host 192.168.1.20
```

### Start Capture via TShark (CLI alternative)
```bash
sudo tshark -i eth0 \
  -f "host 192.168.1.20" \
  -w data/raw/local/normal_traffic.pcap
```

---

## Step-by-Step Capture Procedure

### Step 1 — Normal Traffic (10 minutes)
1. Start Wireshark capture on monitor machine
2. Generate normal traffic from attacker to victim (see nmap_commands.md)
3. Stop capture after 10 minutes
4. Save as: `data/raw/local/normal_traffic.pcap`

### Step 2 — SYN Scan (5 minutes)
1. Start NEW Wireshark capture
2. Run SYN scan from attacker: `nmap -sS 192.168.1.20 -p 1-1000`
3. Stop capture when scan completes
4. Save as: `data/raw/local/syn_scan.pcap`

### Step 3 — TCP Connect Scan (5 minutes)
1. Start NEW Wireshark capture
2. Run TCP scan: `nmap -sT 192.168.1.20 -p 1-1000`
3. Stop capture when scan completes
4. Save as: `data/raw/local/tcp_scan.pcap`

### Step 4 — Xmas Scan (5 minutes)
1. Start NEW Wireshark capture
2. Run Xmas scan: `nmap -sX 192.168.1.20 -p 1-1000`
3. Stop capture when scan completes
4. Save as: `data/raw/local/xmas_scan.pcap`

---

## Verification

After capture, verify each file:
```bash
# Check packet counts
tshark -r data/raw/local/normal_traffic.pcap | wc -l
tshark -r data/raw/local/syn_scan.pcap | wc -l

# Check file sizes
ls -lh data/raw/local/
```

---

## NIC Configuration Notes
- Ensure NIC supports promiscuous mode
- Disable any local firewall rules that may drop scan packets on victim
- On victim machine: `sudo ufw disable` (Ubuntu) or disable Windows Firewall temporarily
- Re-enable firewalls after capture session
