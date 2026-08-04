   
                           
                                               
                                    
   

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.feature_config import FEATURE_NAMES, LABEL_MAP_LOCAL


class TestPcapToCsv:
    def test_synthetic_flows_normal(self):
        from preprocessing.local.pcap_to_csv import _synthetic_flows
        flows = _synthetic_flows("Normal", n=50)
        assert len(flows) == 50
        for f in flows:
            assert f['label'] == "Normal"
            assert f['packet_count'] >= 0
            assert f['syn_flag_count'] >= 0

    def test_synthetic_flows_syn_scan(self):
        from preprocessing.local.pcap_to_csv import _synthetic_flows
        flows = _synthetic_flows("SYN Scan", n=50)
        assert len(flows) == 50
        for f in flows:
            assert f['syn_flag_count'] == 1
            assert f['ack_flag_count'] == 0

    def test_all_features_present(self):
        from preprocessing.local.pcap_to_csv import _synthetic_flows
        flows = _synthetic_flows("Normal", n=10)
        required = [f for f in FEATURE_NAMES]
        for flow in flows:
            for feat in required:
                assert feat in flow, f"Missing feature: {feat}"


class TestCleanFlows:
    def test_removes_short_flows(self):
        from preprocessing.local.clean_flows import clean
                               
        import tempfile
        data = {f: np.random.rand(10) for f in FEATURE_NAMES}
        data['label'] = ['Normal'] * 10
        data['packet_count'] = [1, 1, 5, 10, 3, 2, 8, 1, 6, 4]                        
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df.to_csv(f.name, index=False)
            tmp_in = f.name

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            tmp_out = f.name

        result = clean(input_path=tmp_in, output_path=tmp_out)
        assert len(result) < 10                               
        os.unlink(tmp_in)
        os.unlink(tmp_out)


class TestLabelFlows:
    def test_label_mapping(self):
        assert LABEL_MAP_LOCAL['Normal'] == 0
        assert LABEL_MAP_LOCAL['SYN Scan'] == 1
        assert LABEL_MAP_LOCAL['TCP Connect'] == 2
        assert LABEL_MAP_LOCAL['Xmas Scan'] == 3

    def test_all_classes_mapped(self):
        from preprocessing.local.pcap_to_csv import PCAP_FILES
        for label in PCAP_FILES.values():
            assert label in LABEL_MAP_LOCAL, f"Label not mapped: {label}"


class TestFeatureConfig:
    def test_feature_count(self):
        assert len(FEATURE_NAMES) == 10

    def test_no_duplicates(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_required_features_present(self):
        required = ['syn_flag_count', 'packet_rate', 'flow_duration',
                    'protocol_type', 'dst_port_diversity']
        for f in required:
            assert f in FEATURE_NAMES
