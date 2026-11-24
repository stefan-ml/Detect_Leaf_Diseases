"""Frame -> pretrained leaf segmentation -> leaf ROI -> species -> condition."""
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
LABELS = {'Apple':'Jabuka', 'Corn':'Kukuruz', 'Grape':'Vinova loza', 'Pepper,_bell':'Paprika',
          'Potato':'Krompir', 'Pumpkin':'Tikva', 'Soybean':'Soja', 'Tomato':'Paradajz'}


def decode_frame(data):
    if not data or len(data) > 1_500_000:
        raise ValueError('Kadar je prazan ili veći od 1,5 MB.')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'JPEG' or min(image.size) < 64 or max(image.size) > 1920:
                raise ValueError('Očekuje se JPEG dimenzija 64 do 1920 piksela.')
            return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Neispravan JPEG kadar.') from exc


class SpeciesPipeline:
    def __init__(self, router=None, specialists=None, segmenter=None):
        import torch
        # Small per-frame CPU transforms are faster with a bounded thread pool.
        torch.set_num_threads(4)
        if router is None:
            from inference.bioclip_linear import BioCLIPLinearRouter
            router = BioCLIPLinearRouter(ROOT, ROOT/'models/bioclip_linear')
        self.router = router
        if specialists is None:
            from inference.disease_runtime import DiseaseSpecialists
            specialists = DiseaseSpecialists(ROOT, device=router.verifier.device)
        self.specialists = specialists
        if segmenter is None:
            from inference.leaf_runtime import LeafSegmenter
            segmenter = LeafSegmenter(ROOT, device=router.verifier.device)
        self.segmenter = segmenter
        metadata = getattr(router, 'metadata', {})
        self.manifest = dict(task='plant_species_and_condition', architecture='GenYOLO-Leaf nano segmentation + BioCLIP 2 + accepted-crop ConvNeXt',
            species_task='species_router_bioclip2_linear', specialist_count=len(specialists.metadata),
            input_scope='whole_frame', cpu_threads=torch.get_num_threads(), species_image_encoder_passes=1,
            accepted_frame_image_encoder_passes=3, rejected_frame_image_encoder_passes=2,
            no_leaf_image_encoder_passes=1, segmentation_enabled=True, classifier_input_scope='largest_leaf_roi',
            segmenter=segmenter.metadata, roi_calibration_status='Not recalibrated; exploratory pretrained leaf ROI deployment',
            classifier_sha256=metadata.get('head_sha256'),
            training_signature=metadata.get('selection', {}).get('signature'),
            classes=list(LABELS), species=list(LABELS.values()), disease_prediction=True,
            specialists={crop: m['checkpoint_sha256'] for crop,m in specialists.metadata.items()},
            device=router.verifier.device,
            device_name=torch.cuda.get_device_name() if router.verifier.device.startswith('cuda') else 'CPU',
            project_root=str(ROOT), method=getattr(router,'active_rejection_method',router.rule['method']), preprocessing='GenYOLO-Leaf maske; najveći list sa 10% konteksta; bez maskiranja pozadine klasifikatora')
        if getattr(router,'reference_gate',None):
            self.manifest['reference_rejection'] = router.reference_gate.metadata
            self.manifest['roi_calibration_status'] = 'Combined rule calibrated on public ROI pilot; phone-domain validation pending'
        start = time.perf_counter()
        router.predict(Image.new('RGB', (224, 224), (80, 130, 70)))
        if hasattr(specialists, 'warmup'):
            specialists.warmup()
        if hasattr(segmenter, 'warmup'):
            segmenter.warmup()
        self.manifest['warmup_ms'] = (time.perf_counter()-start)*1000

    def process(self, data, settings=None):
        start = time.perf_counter()
        image = decode_frame(data)
        decoded = time.perf_counter()
        input_size = list(image.size)
        light = np.asarray(image.resize((32, 32))).max(axis=2).mean()
        checked = time.perf_counter()
        preparation = dict(decode_ms=(decoded-start)*1000, light_check_ms=(checked-decoded)*1000)
        if light < 12:
            elapsed = (time.perf_counter()-start)*1000
            return dict(accepted=False, species=None, status='LOW_LIGHT',
                message='Kadar je previše taman. Osvetli list i pokušaj ponovo.',
                species_label=None, condition_class=None, condition_label=None, condition_confidence=None, healthy=None,
                disease_predictions=[], predictions=[], processing_ms=elapsed, input_size=input_size, roi_size=None,
                timings=dict(**preparation, segmentation_ms=0., bioclip_ms=0., head_ms=0., disease_ms=0., encoder_ms=0., taxonomy_ms=0., convnext_ms=0., total_ms=elapsed), disease_prediction=False, overlay=data)
        segmentation = self.segmenter.predict(image)
        detection = {key:segmentation[key] for key in ['leaf_count','leaves','selected_leaf_box']}
        if not segmentation['found']:
            elapsed=(time.perf_counter()-start)*1000
            return dict(accepted=False,species=None,species_label=None,status='NO_LEAF',
                message='List nije pronađen. Usmeri kameru ka listu.',condition_class=None,condition_label=None,
                condition_confidence=None,healthy=None,disease_prediction=False,disease_predictions=[],predictions=[],
                **detection,processing_ms=elapsed,equivalent_fps=1000/elapsed,input_size=input_size,roi_size=None,
                timings=dict(**preparation, **segmentation.get('timings', {}), segmentation_ms=segmentation['segmentation_ms'],bioclip_ms=0.,head_ms=0.,disease_ms=0.,
                    encoder_ms=0.,taxonomy_ms=0.,convnext_ms=0.,total_ms=elapsed),overlay=segmentation['overlay'])
        image = segmentation['roi']
        result = self.router.predict(image)
        result['input_size'] = input_size
        result['roi_size'] = list(image.size)
        result.update(detection)
        result['timings'].update(preparation)
        result['timings'].update(segmentation.get('timings', {}))
        result['timings']['segmentation_ms'] = segmentation['segmentation_ms']
        result.update(condition_class=None, condition_label=None, condition_confidence=None,
                      healthy=None, disease_prediction=False, disease_predictions=[])
        result['timings']['disease_ms'] = 0.
        result['timings']['convnext_ms'] = 0.
        if result['accepted']:
            crop = result['species']
            if crop not in LABELS:
                raise ValueError('Router accepted an unsupported species.')
            result.update(self.specialists.predict(image, crop))
            result['timings']['disease_ms'] = result.pop('disease_ms')
            result['timings']['convnext_ms'] = result['timings']['disease_ms']
        else:
            result['species'] = None
        result['species_label'] = LABELS.get(result['species'])
        result['candidate_label'] = LABELS.get(result['candidate_species'], result['candidate_species'])
        result['bio_candidate_label'] = LABELS.get(result['bioclip_known_candidate'], result['bioclip_known_candidate'])
        result['message'] = ('Podržana biljna vrsta; prikazan je predlog stanja lista.' if result['accepted']
                             else 'Nepoznata ili nesigurna biljka. Dijagnoza bolesti se ne pokreće.')
        result['processing_ms'] = (time.perf_counter()-start)*1000
        result['timings']['total_ms'] = result['processing_ms']
        result['equivalent_fps'] = 1000/result['processing_ms']
        result['predictions'] = []
        result['overlay'] = segmentation['overlay']
        return result

    def benchmark(self, data, repeats=10):
        # Warmups excluded; synchronized per-model timings are provided by the router.
        warmups = [self.process(data) for _ in range(2)]
        values = [self.process(data) for _ in range(repeats)]
        if any(v['status']=='LOW_LIGHT' for v in values):
            raise ValueError('Za merenje izaberi osvetljenu fotografiju lista.')
        total = np.array([v['processing_ms'] for v in values])
        component_timings = {}
        for name in sorted(set().union(*(v['timings'] for v in values))):
            samples = [v['timings'][name] for v in values if name in v['timings']]
            component_timings[name] = dict(mean_ms=float(np.mean(samples)), median_ms=float(np.median(samples)),
                                          p95_ms=float(np.quantile(samples, .95)))
        return dict(repeats=repeats, warmups=2, mean_ms=float(total.mean()), median_ms=float(np.median(total)),
            p95_ms=float(np.quantile(total, .95)), min_ms=float(total.min()), max_ms=float(total.max()),
            equivalent_fps=float(1000/total.mean()),
            bioclip_median_ms=float(np.median([v['timings']['bioclip_ms'] for v in values])),
            head_median_ms=float(np.median([v['timings']['head_ms'] for v in values])),
            disease_median_ms=float(np.median([v['timings']['disease_ms'] for v in values])),
            segmentation_median_ms=float(np.median([v['timings']['segmentation_ms'] for v in values])),
            convnext_median_ms=float(np.median([v['timings']['convnext_ms'] for v in values])),
            component_timings=component_timings,
            samples=[dict(iteration=i+1, status=v['status'], timings=v['timings'], input_size=v.get('input_size'),
                          roi_size=v.get('roi_size')) for i,v in enumerate(values)],
            warmup_samples=[dict(iteration=i+1, status=v['status'], timings=v['timings']) for i,v in enumerate(warmups)],
            device=self.manifest['device_name'],
            note='Ponovljena obrada istog kadra posle zagrevanja; ne uključuje prenos sa kamere.')
