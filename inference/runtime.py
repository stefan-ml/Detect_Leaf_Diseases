"""Portable runtime for the selected frozen BioCLIP 2 + ConvNeXt models."""
from pathlib import Path
import hashlib, json, time, os
import numpy as np
import pandas as pd
import torch, timm, open_clip
from PIL import Image, ImageOps
from torchvision import transforms as T
from filelock import FileLock

KNOWN={'Apple','Corn','Grape','Pepper,_bell','Potato','Pumpkin','Soybean','Tomato'}

def write_json(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    for attempt in range(50):
        try:
            temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
            os.replace(temporary,path); return
        except PermissionError:
            if attempt==49: raise
            time.sleep(.1)

def make_transforms(config,size):
    evaluation=T.Compose([T.Resize((size,size),interpolation=T.InterpolationMode.BICUBIC,antialias=True),
                          T.ToTensor(),T.Normalize(config['mean'],config['std'])])
    return None,evaluation

VERSION = 'bioclip2-full-plant-taxonomy-v1'

MEAN = (.48145466, .4578275, .40821073)

STD = (.26862954, .26130258, .27577711)

CROP_NAMES = {
    'Apple': ['Malus domestica', 'Malus pumila'],
    'Corn': ['Zea mays'], 'Grape': ['Vitis vinifera'],
    'Pepper,_bell': ['Capsicum annuum', 'Solanum annuum'],
    'Potato': ['Solanum tuberosum'],
    'Pumpkin': ['Cucurbita pepo', 'Cucurbita maxima', 'Cucurbita moschata'],
    'Soybean': ['Glycine max'],
    'Tomato': ['Solanum lycopersicum', 'Lycopersicon esculentum'],
}

COMMON_NAMES = {'Apple':'apple', 'Corn':'corn', 'Grape':'grapevine', 'Pepper,_bell':'bell pepper',
                'Potato':'potato', 'Pumpkin':'pumpkin', 'Soybean':'soybean', 'Tomato':'tomato'}

def known_prompts():
    return [(crop, name, [f'a photo of {name}.', f'a photo of a leaf of {name}.',
                         f'a photo of a {COMMON_NAMES[crop]} plant.',
                         f'a close-up photo of a {COMMON_NAMES[crop]} leaf.'])
            for crop, names in CROP_NAMES.items() for name in names]

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def score_methods(table, scales):
    bio = table.bio_margin.to_numpy()
    energy = table.convnext_energy.to_numpy()
    return dict(bioclip_similarity=-table.bio_known_similarity.to_numpy(),
        bioclip_taxonomy_margin=bio,
        bioclip_convnext_gate=np.maximum(
            (bio-scales['bio']['center'])/scales['bio']['scale'],
            (energy-scales['energy']['center'])/scales['energy']['scale']))

_hash=file_hash

def scores_from_logits(logits):
    logits = np.asarray(logits,dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 8 or not np.isfinite(logits).all():
        raise ValueError('Expected finite logits for exactly eight species.')
    maximum = logits.max(axis=1)
    log_normalizer = maximum + np.log(np.exp(logits-maximum[:,None]).sum(axis=1))
    probabilities = np.exp(logits-log_normalizer[:,None])
    return dict(energy=-log_normalizer,confidence=probabilities.max(axis=1),
                predicted=probabilities.argmax(axis=1))

def acceptance(scores,rule):
    return (scores['energy'] <= rule['energy_threshold']) & (scores['confidence'] >= rule['min_confidence'])

class SpeciesRouter:
    """Load once, then route local PIL images to a species specialist or reject."""
    def __init__(self,export_dir,device='auto'):
        directory = Path(export_dir)
        self.metadata = json.loads((directory/'model_manifest.json').read_text())
        self.rule = json.loads((directory/'rejection.json').read_text())
        if (self.metadata.get('task') != 'species_only_known_scope'
            or self.metadata['classes'] != self.rule['classes'] or len(self.rule['classes'])!=8
            or self.metadata['selection']['signature'] != self.rule['signature']
            or self.metadata['rejection_sha256'] != _hash(directory/'rejection.json')
            or self.metadata['checkpoint_sha256'] != _hash(directory/'model_state.pt')
            or self.rule['exported_model_sha256'] != self.metadata['checkpoint_sha256']):
            raise ValueError('Export model, species order or rejection calibration mismatch.')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        self.model = timm.create_model(self.metadata['architecture'],pretrained=False,num_classes=8)
        self.model.load_state_dict(torch.load(directory/'model_state.pt',map_location='cpu',weights_only=True))
        self.model.to(self.device).eval()
        preprocessing = self.metadata['preprocessing']
        _,self.transform = make_transforms(preprocessing,preprocessing['size'])

    @torch.inference_mode()
    def predict(self,image):
        if isinstance(image,(str,Path)):
            with Image.open(image) as opened:
                tensor = self.transform(ImageOps.exif_transpose(opened).convert('RGB'))
        elif isinstance(image,Image.Image):
            tensor = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        else:
            raise TypeError('Expected an image path or PIL image.')
        with torch.autocast(device_type='cuda',enabled=self.device.startswith('cuda')):
            logits = self.model(tensor.unsqueeze(0).to(self.device)).float().cpu().numpy()
        scores = scores_from_logits(logits)
        accepted = bool(acceptance(scores,self.rule)[0])
        return dict(accepted=accepted,species=self.rule['classes'][int(scores['predicted'][0])] if accepted else None,
                    status='KNOWN_SPECIES' if accepted else 'UNCERTAIN',
                    confidence=float(scores['confidence'][0]),energy=float(scores['energy'][0]))
def prepare_catalog(root):
    directory=Path(root)/'models/bioclip2/plant_catalog'
    metadata=json.loads((directory/'catalog.json').read_text())
    if metadata['mapping'] != CROP_NAMES or metadata['version'] != VERSION:
        raise ValueError('Plant catalog mapping/version mismatch.')
    if file_hash(directory/'features.npy') != metadata['features_sha256']:
        raise ValueError('Plant catalog checksum mismatch.')
    return directory


class BioCLIPVerifier:
    def __init__(self, root, device='auto'):
        self.root = Path(root).resolve()
        self.directory = self.root/'models/bioclip2'
        self.catalog_directory = prepare_catalog(self.root)
        self.metadata = json.loads((self.catalog_directory/'catalog.json').read_text())
        if file_hash(self.catalog_directory/'labels.csv') != self.metadata['labels_sha256']:
            raise ValueError('Catalog labels changed.')
        self.labels = pd.read_csv(self.catalog_directory/'labels.csv', keep_default_na=False)
        self.classes = sorted(KNOWN)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        provenance = json.loads((self.directory/'provenance.json').read_text())
        model_spec = next(x for x in provenance['files'] if x['local_name'] == 'open_clip_model.safetensors')
        self.model_hash = file_hash(self.directory/'open_clip_model.safetensors')
        if self.model_hash != model_spec['sha256']:
            raise ValueError('BioCLIP 2 weights changed.')
        config = json.loads((self.directory/'open_clip_config.json').read_text())
        local_config = self.directory/'sir2_bioclip2_local.json'
        local_config.write_text(json.dumps(config['model_cfg']), encoding='utf-8')
        open_clip.add_model_config(local_config)
        print('Loading frozen BioCLIP 2 weights on', self.device, flush=True)
        self.model = open_clip.create_model('sir2_bioclip2_local',
            pretrained=str(self.directory/'open_clip_model.safetensors'), device=self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # The official pybioclip TreeOfLife inference uses whole-image tensor resize.
        self.transform = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True), T.Normalize(MEAN, STD)])
        features = np.load(self.catalog_directory/'features.npy', mmap_mode='r')
        self.text = torch.from_numpy(np.array(features)).to(self.device)
        # Some official catalog columns are zero, including potato/grape. Encode ALL
        # known crops with the same fixed prompt protocol instead of uneven repairs.
        prompts = known_prompts()
        proto_path = self.catalog_directory/'known_crop_text.npy'
        meta_path = proto_path.with_suffix('.json')
        spec = dict(model_sha256=self.model_hash, prompts=prompts)
        with FileLock(self.directory/'.known_text.lock', timeout=0):
            if meta_path.exists():
                saved = json.loads(meta_path.read_text())
                if (json.dumps(saved['spec'], sort_keys=True) != json.dumps(spec, sort_keys=True)
                        or file_hash(proto_path) != saved['file_sha256']):
                    raise ValueError('Known crop text prototypes changed.')
                self.known_text = torch.from_numpy(np.load(proto_path)).to(self.device)
            else:
                tokenizer = open_clip.get_tokenizer('sir2_bioclip2_local')
                text_rows = []
                with torch.inference_mode():
                    for crop, scientific, texts in prompts:
                        tokens = tokenizer(texts).to(self.device)
                        with torch.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
                            encoded = self.model.encode_text(tokens)
                        encoded = torch.nn.functional.normalize(encoded.float(), dim=-1).mean(dim=0)
                        text_rows.append(torch.nn.functional.normalize(encoded, dim=0))
                self.known_text = torch.stack(text_rows)
                np.save(proto_path, self.known_text.cpu().numpy())
                write_json(meta_path, dict(spec=spec, file_sha256=file_hash(proto_path)))
        self.crop_indices = [torch.as_tensor([i for i, row in enumerate(prompts) if row[0] == crop], device=self.device)
                             for crop in self.classes]
        self.known_text_hash = hashlib.sha256(self.known_text.cpu().numpy().tobytes()).hexdigest()
        self.other_mask = torch.as_tensor(self.labels.crop_class.eq('').to_numpy(), device=self.device)
        self.preprocessing = dict(size=224, resize='whole-image float32 bilinear antialias', mean=MEAN, std=STD,
                                  exif_transpose=True)
        torch.backends.cuda.matmul.allow_tf32 = False

    @torch.inference_mode()
    def score_features(self, embedding):
        embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        known = torch.stack([(embedding @ self.known_text[indices].T).max(dim=1).values for indices in self.crop_indices], dim=1)
        best_other = torch.full((len(embedding),), -float('inf'), device=self.device)
        best_index = torch.zeros(len(embedding), dtype=torch.long, device=self.device)
        for start in range(0, len(self.text), 8192):
            stop = min(start+8192, len(self.text))
            similarities = embedding @ self.text[start:stop].T
            similarities[:, ~self.other_mask[start:stop]] = -float('inf')
            score, index = similarities.max(dim=1)
            improved = score > best_other
            best_other = torch.maximum(best_other, score)
            best_index = torch.where(improved, index+start, best_index)
        values, candidate = known.max(dim=1)
        return torch.stack([values, best_other, best_other-values, candidate.float(), best_index.float()], dim=1).cpu().numpy()

    @torch.inference_mode()
    def score_batch(self, tensor):
        with torch.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
            features = self.model.encode_image(tensor.to(self.device, non_blocking=True))
        return self.score_features(features)

class BioCLIPSpeciesRouter:
    def __init__(self, root, device='auto', verifier=None):
        self.root = Path(root).resolve()
        run = self.root/'models'
        self.rule = json.loads((run/'router/rule.json').read_text())
        if self.rule['smoke'] or not self.rule['unknown_detection_evaluated']:
            raise ValueError('Run full BioCLIP calibration and evaluation first.')
        self.verifier = verifier if verifier is not None else BioCLIPVerifier(self.root, device)
        self.classifier = SpeciesRouter(run/'convnext', device=self.verifier.device)
        if (self.rule['bioclip_model_sha256'] != self.verifier.model_hash
            or self.rule['plant_features_sha256'] != self.verifier.metadata['features_sha256']
            or self.rule['known_text_sha256'] != self.verifier.known_text_hash
            or self.rule['version'] != VERSION
            or self.rule['convnext_model_sha256'] != self.classifier.metadata['checkpoint_sha256']
            or self.rule['mapping'] != CROP_NAMES):
            raise ValueError('BioCLIP/ConvNeXt model or taxonomy mismatch.')

    def predict(self, image):
        started = time.perf_counter()
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                rgb = ImageOps.exif_transpose(opened).convert('RGB')
        elif isinstance(image, Image.Image):
            rgb = ImageOps.exif_transpose(image).convert('RGB')
        else:
            raise TypeError('Expected image path or PIL image.')
        def sync():
            if self.verifier.device.startswith('cuda'):
                torch.cuda.synchronize(self.verifier.device)
        sync()
        bio_start = time.perf_counter()
        bio = self.verifier.score_batch(self.verifier.transform(rgb).unsqueeze(0))[0]
        sync()
        bio_end = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=self.verifier.device.startswith('cuda')):
            logits = self.classifier.model(self.classifier.transform(rgb).unsqueeze(0).to(self.verifier.device))
        scores = scores_from_logits(logits.float().cpu().numpy())
        sync()
        cnn_end = time.perf_counter()
        table = pd.DataFrame([dict(bio_known_similarity=bio[0], bio_margin=bio[2], convnext_energy=float(scores['energy'][0]))])
        score = float(score_methods(table, self.rule['scales'])[self.rule['method']][0])
        accepted = bool(score <= self.rule['methods'][self.rule['method']]['threshold'])
        candidate = self.classifier.rule['classes'][int(scores['predicted'][0])]
        return dict(accepted=accepted, species=candidate if accepted else None,
            status='KNOWN_SPECIES' if accepted else 'UNKNOWN_OR_UNCERTAIN', candidate_species=candidate,
            bioclip_known_candidate=self.verifier.classes[int(bio[3])],
            bioclip_other_candidate=str(self.verifier.labels.iloc[int(bio[4])].scientific_name),
            bioclip_known_similarity=float(bio[0]), bioclip_other_similarity=float(bio[1]),
            score=score, threshold=self.rule['methods'][self.rule['method']]['threshold'], method=self.rule['method'],
            confidence=float(scores['confidence'][0]),
            timings=dict(bioclip_ms=(bio_end-bio_start)*1000, convnext_ms=(cnn_end-bio_end)*1000,
                         decision_ms=(time.perf_counter()-cnn_end)*1000, total_ms=(time.perf_counter()-started)*1000),
            disease_prediction=False)
