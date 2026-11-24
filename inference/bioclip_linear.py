"""One frozen BioCLIP image pass, trained crop head and taxonomy rejection."""
import json
from pathlib import Path
import time

import torch
from torch import nn
from PIL import Image, ImageOps

from inference.runtime import BioCLIPVerifier, KNOWN, file_hash, scores_from_logits


class BioCLIPLinearRouter:
    def __init__(self, root, bundle, device='auto', verifier=None):
        self.root=Path(root).resolve(); self.bundle=Path(bundle).resolve()
        self.metadata=json.loads((self.bundle/'manifest.json').read_text())
        self.rule=json.loads((self.bundle/'rejection.json').read_text())
        m=self.metadata
        if (m['task']!='frozen_bioclip2_linear_species_open_set' or m['classes']!=sorted(KNOWN)
            or not m['backbone_frozen'] or self.rule['method']!='bioclip_taxonomy_margin'
            or m['head_sha256']!=file_hash(self.bundle/'head.pt')
            or m['rejection_sha256']!=file_hash(self.bundle/'rejection.json')
            or m['rejection_threshold']!=self.rule['methods']['bioclip_taxonomy_margin']['threshold']):
            raise ValueError('Linear-probe bundle, class order or rejection mismatch.')
        self.verifier=verifier if verifier is not None else BioCLIPVerifier(self.root,device)
        v=self.verifier
        if (m['encoder_sha256']!=v.model_hash or m['catalog_sha256']!=v.metadata['features_sha256']
            or m['labels_sha256']!=v.metadata['labels_sha256'] or m['known_text_sha256']!=v.known_text_hash
            or json.dumps(m['preprocessing'],sort_keys=True)!=json.dumps(v.preprocessing,sort_keys=True)):
            raise ValueError('Frozen encoder, reference catalog or preprocessing mismatch.')
        self.head=nn.Linear(m['feature_dim'],8).to(v.device)
        self.head.load_state_dict(torch.load(self.bundle/'head.pt',map_location=v.device,weights_only=True))
        self.head.eval()
        self.reference_gate = None
        if (self.root/'models/roi_rejection/manifest.json').exists():
            from inference.reference_gate import ReferenceGate
            self.reference_gate = ReferenceGate(self.root, self.metadata)
        self.active_rejection_method = ('bioclip_taxonomy_and_reference_knn' if self.reference_gate
                                        else 'bioclip_taxonomy_margin')

    def sync(self):
        if self.verifier.device.startswith('cuda'):
            torch.cuda.synchronize(self.verifier.device)

    @torch.inference_mode()
    def predict(self,image):
        if isinstance(image,(str,Path)):
            with Image.open(image) as opened:
                rgb=ImageOps.exif_transpose(opened).convert('RGB')
        elif isinstance(image,Image.Image):
            rgb=ImageOps.exif_transpose(image).convert('RGB')
        else:
            raise TypeError('Expected a local image path or PIL image.')
        self.sync(); begun=time.perf_counter()
        tensor=self.verifier.transform(rgb).unsqueeze(0).to(self.verifier.device)
        with torch.autocast(device_type='cuda',enabled=self.verifier.device.startswith('cuda')):
            features=self.verifier.model.encode_image(tensor)
        features=nn.functional.normalize(features.float(),dim=-1)
        self.sync(); encoded=time.perf_counter()
        bio=self.verifier.score_features(features)[0]
        self.sync(); verified=time.perf_counter()
        # FP32 head, identical to probe training; no second image encoder.
        logits=self.head(features).float().cpu().numpy()
        scores=scores_from_logits(logits)
        accepted=bool(bio[2]<=self.metadata['rejection_threshold'])
        candidate=self.metadata['classes'][int(scores['predicted'][0])]
        headed=time.perf_counter()
        gate = getattr(self,'reference_gate',None)
        detail = gate.decide(features.cpu().numpy()[0],candidate,float(bio[2])) if gate else {}
        if gate:
            accepted=detail['accepted']
        self.sync(); classified=time.perf_counter()
        result = dict(accepted=accepted,species=candidate if accepted else None,
            status='KNOWN_SPECIES' if accepted else 'UNKNOWN_OR_UNCERTAIN',candidate_species=candidate,
            bioclip_known_candidate=self.verifier.classes[int(bio[3])],
            bioclip_other_candidate=str(self.verifier.labels.iloc[int(bio[4])].scientific_name),
            bioclip_known_similarity=float(bio[0]),bioclip_other_similarity=float(bio[1]),
            score=float(bio[2]),threshold=self.metadata['rejection_threshold'],method='bioclip_taxonomy_margin',
            confidence=float(scores['confidence'][0]),disease_prediction=False,
            architecture='Frozen BioCLIP 2 + trained eight-crop linear head',
            timings=dict(encoder_ms=(encoded-begun)*1000,taxonomy_ms=(verified-encoded)*1000,
                linear_head_ms=(headed-verified)*1000, reference_gate_ms=(classified-headed)*1000,
                head_ms=(classified-verified)*1000,bioclip_ms=(verified-begun)*1000,convnext_ms=0.,
                total_ms=(classified-begun)*1000))
        result.update(detail)
        return result
