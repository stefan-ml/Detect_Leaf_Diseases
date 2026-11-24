"""Verified frozen-feature reference rejection, sharing the BioCLIP image pass."""
import json
from pathlib import Path
import numpy as np
from inference.runtime import file_hash


class ReferenceGate:
    def __init__(self, root, head_metadata):
        root=Path(root);bundle=root/'models/roi_rejection'
        m=json.loads((bundle/'manifest.json').read_text())
        identity=m['feature_identity']
        for key in ['head_sha256','encoder_sha256','catalog_sha256','known_text_sha256','classes']:
            if identity[key]!=head_metadata[key]:
                raise ValueError('Reference gate identity mismatch: '+key)
        if file_hash(root/'models/leaf_segmenter/manifest.json')!=identity['segmenter_manifest_sha256']:
            raise ValueError('Reference gate segmentation mismatch.')
        if file_hash(bundle/'reference_bank.npz')!=m['reference_sha256']:
            raise ValueError('Reference gate bank checksum mismatch.')
        if m['rule']['method']!='combined' or m['k']!=5:
            raise ValueError('Expected evaluated combined reference gate.')
        if not all(np.isfinite(m['rule'][k]) for k in ['taxonomy_threshold','knn_threshold']):
            raise ValueError('Invalid reference gate thresholds.')
        with np.load(bundle/'reference_bank.npz',allow_pickle=False) as bank:
            features=bank['features'].copy();labels=bank['labels'].copy()
        if (features.ndim!=2 or features.shape!=(len(labels),head_metadata['feature_dim'])
            or not np.isfinite(features).all() or set(labels)!=set(head_metadata['classes'])
            or not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-4)):
            raise ValueError('Invalid normalized reference bank or class labels.')
        self.banks={crop:features[labels==crop] for crop in head_metadata['classes']}
        if any(len(bank)<m['k'] for bank in self.banks.values()):
            raise ValueError('Insufficient crop reference images.')
        self.metadata=m

    def decide(self, feature, candidate, margin):
        bank=self.banks[candidate]
        similarity=np.asarray(feature,dtype=np.float32).reshape(1,-1)@bank.T
        distance=float(1-np.partition(similarity,-self.metadata['k'],axis=1)[:,-self.metadata['k']:].mean())
        rule=self.metadata['rule']
        taxonomy_pass=bool(margin<=rule['taxonomy_threshold'])
        reference_pass=bool(distance<=rule['knn_threshold'])
        return dict(accepted=taxonomy_pass and reference_pass,
            method='bioclip_taxonomy_and_reference_knn',threshold=rule['taxonomy_threshold'],
            reference_distance=distance,reference_threshold=rule['knn_threshold'],
            taxonomy_pass=taxonomy_pass,reference_pass=reference_pass,
            reference_images=len(bank),reference_k=self.metadata['k'])
