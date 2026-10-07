"""Export the FIBER_CENTRE_7624_EMA backbone as an nnU-Net model folder.

The folder uses the layout of scrollprize/fiber_hz_vt (checkpoint_final.pth,
plans.json and dataset.json at the root). Only the four-class segmentation
network is exported; the centre and affinity heads were used for training only.
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXPORTS = HERE.parent
SOURCE = EXPORTS/'pherc0175a-full-7624/weights/FIBER_CENTRE_7624_EMA.pt'
SOURCE_SHA256 = '144a33b154c440ab4990bf05953e454a6cd8167cd28a8a846746a64cfd605498'
OFFICIAL = EXPORTS/'pherc1203highres_x10103_y12566_z5060_l0/cube-ct/model-test/hz-vt'
OUT = HERE/'upload'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if sha256(SOURCE) != SOURCE_SHA256:
        raise SystemExit(f'{SOURCE.name}: unexpected SHA-256')
    plans = json.loads((OFFICIAL/'plans.json').read_text())
    dataset = json.loads((OFFICIAL/'dataset.json').read_text())
    # The model was trained on 128-voxel crops; the architecture does not depend on it.
    plans['configurations']['3d_fullres']['patch_size'] = [128, 128, 128]
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    backbone = {k.removeprefix('backbone.'): v for k, v in source['model'].items() if k.startswith('backbone.')}
    reference = torch.load(OFFICIAL/'checkpoint_final.pth', map_location='cpu', weights_only=False, mmap=True)['network_weights']
    if set(backbone) != set(reference) or any(backbone[k].shape != reference[k].shape for k in reference):
        raise SystemExit('The backbone does not match the fiber_hz_vt architecture')
    OUT.mkdir(exist_ok=True)
    torch.save({
        'network_weights': OrderedDict((k, backbone[k]) for k in reference),
        'init_args': {'plans': plans, 'configuration': '3d_fullres', 'fold': 0,
                      'dataset_json': dataset, 'unpack_dataset': True, 'device': 'cuda'},
        'trainer_name': 'nnUNetTrainer',
        'inference_allowed_mirroring_axes': (0, 1, 2),
    }, OUT/'checkpoint_final.pth')
    (OUT/'plans.json').write_text(json.dumps(plans, indent=4) + '\n')
    (OUT/'dataset.json').write_text((OFFICIAL/'dataset.json').read_text())
    print(json.dumps({'checkpoint_final.pth': sha256(OUT/'checkpoint_final.pth'),
                      'bytes': (OUT/'checkpoint_final.pth').stat().st_size}, indent=1))


if __name__ == '__main__':
    main()
