"""CPU verification of public backbone tensors; optional trusted historical source.

No download, CT access, model execution or GPU use. Source checkpoint loading is
allowed only after matching its recorded full-file SHA-256.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SHA = '6837f3fe8fca123e6ed0b91bb052ce5218ca6eb6889800e4d578b6c8aaf59767'
SOURCE_SHA = 'f0e8caf18cf4c4f5c576d0247fe450294c210bf0171f39517a1fe08f77dae4e9'


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tensor_record(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True, help='Local public checkpoint_final.pth')
    parser.add_argument('--source', type=Path, help='Optional original V6 last.pt with known SHA')
    parser.add_argument('--write-manifest', action='store_true', help='Publication preparation only')
    args = parser.parse_args()
    if file_digest(args.checkpoint) != PUBLIC_SHA:
        raise SystemExit('Public checkpoint SHA-256 mismatch')
    public = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    records = {k: tensor_record(v) for k,v in public['network_weights'].items()}
    source_verified = False
    if args.source:
        if file_digest(args.source) != SOURCE_SHA:
            raise SystemExit('Historical source checkpoint SHA-256 mismatch')
        source = torch.load(args.source, map_location='cpu', weights_only=False, mmap=True)
        if source['step'] != 7624:
            raise SystemExit('Unexpected source step')
        config_path = ROOT / 'original/fiber-centre-compression-v6/runs/multiscroll-centre-v6/config.json'
        if source['config'] != json.loads(config_path.read_text()):
            raise SystemExit('Embedded source config mismatch')
        backbone = {k.removeprefix('backbone.'):v for k,v in source['ema'].items() if k.startswith('backbone.')}
        if set(backbone) != set(records):
            raise SystemExit('Source/export key mismatch')
        for name,value in backbone.items():
            if tensor_record(value) != records[name]:
                raise SystemExit('Source EMA/export tensor mismatch: '+name)
        source_verified = True
        if args.write_manifest:
            (ROOT / 'provenance/checkpoint-embedded-config.json').write_text(json.dumps(source['config'],indent=2)+'\n')
    manifest = {'publicCheckpointSHA256':PUBLIC_SHA,'sourceCheckpointSHA256':SOURCE_SHA,
                'sourceStep':7624,'sourceState':'ema.backbone','tensorEncoding':'Contiguous native torch bytes on little-endian host; shape and dtype recorded separately',
                'sourceEMAMatchedAtManifestCreation':source_verified,'tensors':records}
    path = ROOT / 'provenance/backbone-tensors.json'
    if args.write_manifest:
        if not source_verified:
            raise SystemExit('Writing a source/export manifest requires --source')
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    elif records != json.loads(path.read_text())['tensors']:
        raise SystemExit('Public tensors differ from the archived manifest')
    print(json.dumps({'publicFileSHA256Verified':True,'backboneTensorsVerified':len(records),
                      'sourceEMAVerifiedThisInvocation':source_verified,
                      'scope':'Byte identity only; no performance or label-accuracy claim'},indent=2))


if __name__ == '__main__':
    main()
