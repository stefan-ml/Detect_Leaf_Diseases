"""Pretrained GenYOLO-Leaf nano instance segmentation; local safetensors only."""
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class LeafSegmenter:
    def __init__(self, root, device='cpu'):
        import torch
        from safetensors.torch import load_file
        from ultralytics import YOLO
        from ultralytics.nn.tasks import SegmentationModel
        bundle=Path(root)/'models/leaf_segmenter'
        self.metadata=json.loads((bundle/'manifest.json').read_text())
        weights=bundle/'model.safetensors'
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=self.metadata['weights_sha256']:
            raise ValueError('Leaf segmentation weights do not match their manifest.')
        if self.metadata['names'] != {'0':'Leaf'} or self.metadata['architecture']['nc'] != 1:
            raise ValueError('Expected one leaf segmentation class.')
        model=SegmentationModel(self.metadata['architecture'],verbose=False)
        model.load_state_dict(load_file(str(weights)))
        model.names={0:'Leaf'}
        model.eval()
        self.model=YOLO('yolo11n-seg.yaml',verbose=False)
        self.model.model=model
        self.model.task='segment'
        self.device=device
        self.torch=torch

    def sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize(self.device)

    def warmup(self):
        self.predict(Image.new('RGB',(640,480),(80,130,70)))

    def predict(self, image):
        self.sync();start=time.perf_counter()
        result=self.model.predict(image,imgsz=self.metadata['imgsz'],conf=self.metadata['confidence'],
                                  device=self.device,verbose=False,retina_masks=False,max_det=30)[0]
        self.sync(); inferred=time.perf_counter()
        leaves=[]
        if result.masks is not None:
            for confidence,polygon in zip(result.boxes.conf.cpu().tolist(),result.masks.xy):
                polygon=np.asarray(polygon,dtype=np.float64)
                if len(polygon)<3 or not np.isfinite(polygon).all():
                    continue
                polygon[:,0]=polygon[:,0].clip(0,image.width-1)
                polygon[:,1]=polygon[:,1].clip(0,image.height-1)
                area=abs(np.dot(polygon[:,0],np.roll(polygon[:,1],1))-
                         np.dot(polygon[:,1],np.roll(polygon[:,0],1)))/2
                if area/(image.width*image.height)<self.metadata['min_mask_fraction']:
                    continue
                leaves.append(dict(confidence=float(confidence),area=float(area),polygon=polygon.tolist()))
        leaves.sort(key=lambda leaf:leaf['area'],reverse=True)
        roi=None;selected_box=None
        overlay=image.copy()
        if leaves:
            mask=Image.new('L',image.size)
            mask_draw=ImageDraw.Draw(mask)
            for leaf in leaves:
                mask_draw.polygon([tuple(p) for p in leaf['polygon']],fill=70)
            overlay=Image.composite(Image.new('RGB',image.size,(30,200,100)),overlay,mask)
            draw=ImageDraw.Draw(overlay)
            for leaf in leaves:
                polygon=[tuple(p) for p in leaf['polygon']]
                draw.line(polygon+[polygon[0]],fill=(30,230,100),width=max(2,image.width//400))
            polygon=np.array(leaves[0]['polygon'])
            x1,y1=polygon.min(axis=0);x2,y2=polygon.max(axis=0)
            px=(x2-x1)*.1;py=(y2-y1)*.1
            selected_box=[max(0,int(x1-px)),max(0,int(y1-py)),
                          min(image.width,int(np.ceil(x2+px))+1),min(image.height,int(np.ceil(y2+py))+1)]
            roi=image.crop(selected_box)
            draw.rectangle(selected_box,outline=(255,210,50),width=max(2,image.width//400))
        buffer=io.BytesIO();overlay.save(buffer,format='JPEG',quality=90)
        self.sync()
        finished=time.perf_counter()
        return dict(found=bool(leaves),roi=roi,overlay=buffer.getvalue(),
                    leaf_count=len(leaves),leaves=leaves,selected_leaf_box=selected_box,
                    segmentation_ms=(finished-start)*1000,
                    timings=dict(segmentation_inference_ms=(inferred-start)*1000,
                                 segmentation_postprocess_ms=(finished-inferred)*1000))
