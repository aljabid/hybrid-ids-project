   
                  
                                                                  
   

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestClassifierTraining:

    def test_decision_tree_trains(self):
        from sklearn.tree import DecisionTreeClassifier
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 4, 100)
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X, y)
        assert clf.get_depth() > 0

    def test_predict_proba_shape(self):
        from sklearn.tree import DecisionTreeClassifier
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 4, 100)
        X_test  = np.random.rand(20, 10)
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        assert proba.shape == (20, 4)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_confidence_extraction(self):
        from utils.confidence_utils import get_confidence
        from sklearn.tree import DecisionTreeClassifier
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 4, 100)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        X_single = np.random.rand(10)
        label, conf = get_confidence(clf, X_single)
        assert 0.0 <= conf <= 1.0
        assert label in [0, 1, 2, 3]


class TestConfidenceRouter:

    def _make_mocks(self):
                                                                          
        from sklearn.tree import DecisionTreeClassifier
        import torch
        from models.autoencoder.autoencoder_model import Autoencoder

        X_train = np.random.rand(200, 10)
        y_train = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        ae = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        ae.eval()

        return clf, ae

    def test_router_returns_valid_label(self):
        from hybrid.confidence_router import route
        clf, ae = self._make_mocks()
        x = np.random.rand(10)
        label, module, meta = route(x, clf, ae, ae_threshold=0.05,
                                     confidence_threshold=0.85)
        assert module in ['classifier', 'autoencoder']
        assert isinstance(label, str)
        assert 'confidence' in meta

    def test_router_batch(self):
        from hybrid.confidence_router import route_batch
        clf, ae = self._make_mocks()
        X = np.random.rand(50, 10)
        labels, modules, metas = route_batch(
            X, clf, ae, ae_threshold=0.05, confidence_threshold=0.85)
        assert len(labels) == 50
        assert len(modules) == 50
        assert all(m in ['classifier', 'autoencoder'] for m in modules)

    def test_high_confidence_uses_classifier(self):
                                                                                       
        from sklearn.tree import DecisionTreeClassifier
        from hybrid.confidence_router import route
        from models.autoencoder.autoencoder_model import Autoencoder

                                                                
        X_train = np.vstack([
            np.zeros((50, 10)),
            np.ones((50, 10)) * 2,
            np.ones((50, 10)) * 5,
            np.ones((50, 10)) * 8,
        ])
        y_train = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        ae = Autoencoder(input_dim=10, hidden_dims=[64, 32, 16])
        ae.eval()

        x_attack = np.ones(10) * 2.0
        label, module, meta = route(x_attack, clf, ae,
                                     ae_threshold=0.05,
                                     confidence_threshold=0.85)
        assert meta['confidence'] >= 0.85
        assert module == 'classifier'
