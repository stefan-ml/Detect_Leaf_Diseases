"""Verified eight-crop specialist registry; execute only the accepted crop model."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import timm

from inference.runtime import KNOWN,file_hash,make_transforms

CONDITION_LABELS={
    'healthy':'Zdrav list','Apple_scab':'Čađava krastavost','Black_rot':'Crna trulež',
    'Cedar_apple_rust':'Rđa jabuke','Common_rust':'Rđa kukuruza',
    'Cercospora_leaf_spot Gray_leaf_spot':'Siva pegavost lista (Cercospora)',
    'Northern_Leaf_Blight':'Severna palež lista','Esca_(Black_Measles)':'Eska',
    'Leaf_blight_(Isariopsis_Leaf_Spot)':'Pegavost lista (Isariopsis)',
    'Bacterial_spot':'Bakterijska pegavost','Cercospora_leaf_spot':'Cerkosporna pegavost',
    'Leaf_curl':'Uvijenost lista','Powdery_mildew':'Pepelnica',
    'Bacterial_soft_rot':'Bakterijska meka trulež','Early_blight':'Crna pegavost (Alternaria)',
    'Late_blight':'Plamenjača','Leafroll_virus':'Virus uvijenosti lista',
    'Potato_virus_X':'Virus krompira X (PVX)','Potato_virus_Y':'Virus krompira Y (PVY)',
    'Bacterial_leaf_spot':'Bakterijska pegavost lista','Downy_mildew':'Plamenjača',
    'Mosaic_disease':'Mozaik','Bacterial_blight':'Bakterijska plamenjača',
    'Cercospora_leaf_blight':'Cerkosporna palež lista','Rust':'Rđa',
    'Sudden_death_syndrome':'Sindrom iznenadnog odumiranja','Leaf_Mold':'Lisna plesan',
    'Septoria_leaf_spot':'Septorijska pegavost','Spider_mites Two-spotted_spider_mite':'Oštećenje od grinja',
    'Target_Spot':'Koncentrična pegavost','Tomato_Yellow_Leaf_Curl_Virus':'Virus žute uvijenosti lista',
    'Tomato_mosaic_virus':'Virus mozaika paradajza'}


def condition_label(name):
    condition=name.split('___',1)[1]
    return CONDITION_LABELS.get(condition,condition.replace('_',' '))


class DiseaseSpecialists:
    def __init__(self,root,device='cpu'):
        self.root=Path(root).resolve();self.device=device
        self.directory=self.root/'models/disease_specialists'
        index=json.loads((self.directory/'index.json').read_text())
        if index['task']!='eight_crop_condition_specialists' or set(index['models'])!=KNOWN:
            raise ValueError('Expected exactly eight deployed crop specialists.')
        self.models={};self.transforms={};self.metadata={}
        for crop,spec in index['models'].items():
            directory=(self.directory/spec['directory']).resolve()
            if not directory.is_relative_to(self.directory):
                raise ValueError('Specialist path outside deployment directory.')
            manifest=directory/'model_manifest.json'
            if file_hash(manifest)!=spec['manifest_sha256']:
                raise ValueError('Specialist metadata checksum mismatch: '+crop)
            m=json.loads(manifest.read_text())
            if (m['task']!='crop_condition_specialist' or m['species']!=crop
                or m['selection']['smoke'] or not m['classes'] or len(set(m['classes']))!=len(m['classes'])
                or any(not c.startswith(crop+'___') for c in m['classes'])
                or m['healthy_class']!=crop+'___healthy' or m['healthy_class'] not in m['classes']
                or m['checkpoint_sha256']!=spec['checkpoint_sha256']
                or file_hash(directory/'model_state.pt')!=m['checkpoint_sha256']
                or m['preprocessing']['resize']!='whole-image bicubic'):
                raise ValueError('Specialist species, classes, weights or preprocessing mismatch: '+crop)
            model=timm.create_model(m['architecture'],pretrained=False,num_classes=len(m['classes']))
            model.load_state_dict(torch.load(directory/'model_state.pt',map_location='cpu',weights_only=True))
            for parameter in model.parameters():parameter.requires_grad_(False)
            self.models[crop]=model.to(device).eval()
            _,self.transforms[crop]=make_transforms(m['preprocessing'],m['preprocessing']['size'])
            self.metadata[crop]=m

    def sync(self):
        if self.device.startswith('cuda'):torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def warmup(self):
        for crop,model in self.models.items():
            size=self.metadata[crop]['preprocessing']['size']
            with torch.autocast(device_type='cuda',enabled=self.device.startswith('cuda')):
                model(torch.zeros(1,3,size,size,device=self.device))
        self.sync()

    @torch.inference_mode()
    def predict(self,image,crop):
        if crop not in self.models:
            raise ValueError('No specialist for accepted crop: '+str(crop))
        self.sync();begun=time.perf_counter()
        tensor=self.transforms[crop](image).unsqueeze(0).to(self.device)
        with torch.autocast(device_type='cuda',enabled=self.device.startswith('cuda')):
            logits=self.models[crop](tensor)
        probabilities=logits.float().softmax(1)[0].cpu().numpy()
        self.sync();elapsed=(time.perf_counter()-begun)*1000
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite condition prediction.')
        m=self.metadata[crop];indices=probabilities.argsort()[::-1]
        predicted=m['classes'][int(indices[0])]
        return dict(condition_class=predicted,condition_label=condition_label(predicted),
            condition_confidence=float(probabilities[indices[0]]),healthy=predicted==m['healthy_class'],
            disease_prediction=True,specialist_species=crop,specialist_sha256=m['checkpoint_sha256'],
            unknown_disease_detection_evaluated=False,disease_ms=elapsed,
            disease_predictions=[dict(class_name=m['classes'][int(i)],label=condition_label(m['classes'][int(i)]),
                score=float(probabilities[i])) for i in indices[:3]])
