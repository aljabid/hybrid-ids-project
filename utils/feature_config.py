   
                 
                                                                         
                                                                             
   

FEATURE_NAMES = [
    "flow_duration",
    "packet_count",
    "byte_count",
    "packet_rate",
    "inter_arrival_mean",
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "dst_port_diversity",
    "protocol_type"
]

NUM_FEATURES = len(FEATURE_NAMES)

                                        
CLASS_NAMES_LOCAL = ["Normal", "SYN Scan", "TCP Connect", "Xmas Scan"]
NUM_CLASSES_LOCAL = len(CLASS_NAMES_LOCAL)
LABEL_MAP_LOCAL = {
    "Normal":       0,
    "SYN Scan":     1,
    "TCP Connect":  2,
    "Xmas Scan":    3
}
LABEL_MAP_LOCAL_INV = {v: k for k, v in LABEL_MAP_LOCAL.items()}

                                                              
CLASS_NAMES_BINARY = ["Normal", "PortScan"]
LABEL_MAP_BINARY = {"Normal": 0, "PortScan": 1}
LABEL_MAP_BINARY_INV = {0: "Normal", 1: "PortScan"}

                   
PROTOCOL_MAP = {"TCP": 0, "UDP": 1, "ICMP": 2}
