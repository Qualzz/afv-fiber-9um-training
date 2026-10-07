"""Verify historical config hashes, receipts and public metadata; no ML imports."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing literal '+name)


def main():
    errors=[];checks=0;core=0
    def equal(actual, expected, description):
        nonlocal checks
        checks+=1
        if actual!=expected:errors.append(description)
    inventory=json.loads((ROOT/'provenance/source-inventory.json').read_text())
    for entry in inventory['files']:
        path=ROOT/entry['path']
        if not path.is_file():errors.append('Missing '+entry['path']);continue
        equal(path.stat().st_size,entry['bytes'],'Size: '+entry['path'])
        equal(digest(path),entry['sha256'],'Inventory SHA: '+entry['path'])
    phases=[('fiber-compression-pherc0358-v3','full'),('fiber-compression-multiscroll-v4','multiscroll'),
            ('fiber-full-8000-v5','multiscroll-full-v5'),('fiber-centre-compression-v6','multiscroll-centre-v6')]
    configs={}
    for phase,name in phases:
        folder=ROOT/'original'/phase;run=folder/'runs'/name
        config=json.loads((run/'config.json').read_text());configs[phase]=config
        for filename,expected in config['codeSHA256'].items():
            equal(digest(folder/filename),expected,'Recorded core SHA: '+phase+'/'+filename);core+=1
        expected=config.get('snapshotSHA256',config.get('sourceSnapshotSHA256'))
        equal(digest(run/'snapshot.json'),expected,'Recorded snapshot: '+phase)
        if 'validationSHA256' in config:
            equal(digest(run/'validation.json'),config['validationSHA256'],'Recorded validation: '+phase)
        if (run/'prepared.json').is_file():
            for key,expected in json.loads((run/'prepared.json').read_text()).items():
                if key in ('snapshot','validation','snapshot.json','validation.json'):
                    equal(digest(run/(key if key.endswith('.json') else key+'.json')),expected,'Prepared '+phase+'/'+key)
    v3=ROOT/'original/fiber-compression-pherc0358-v3'
    for name in ('initial','expanded-01'):
        run=ROOT/'provenance/pre-v3'/name;config=json.loads((run/'config.json').read_text())
        for filename,expected in config['codeSHA256'].items():
            equal(digest(v3/filename),expected,'Pre-V3 core: '+name+'/'+filename);core+=1
        equal(digest(run/'snapshot.json'),config['snapshotSHA256'],'Pre-V3 snapshot '+name)
    phase=json.loads((v3/'runs/full/phase.json').read_text())
    equal(digest(v3/'runs/full/validation.json'),phase['validationSHA256'],'V3 phase validation')
    expanded=ROOT/'provenance/pre-v3/expanded-01'
    prephase=json.loads((expanded/'phase.json').read_text())
    equal(digest(expanded/'validation.json'),prephase['validationSHA256'],'Expanded-01 phase validation')
    equal(digest(ROOT/'provenance/pre-v3/initial/validation.json'),prephase['validationSHA256'],'Initial reused validation')
    training_doc=(ROOT/'docs/TRAINING.md').read_text()
    for name,expected in [('Initial best EMA',prephase['parentCheckpointSHA256']),
                          ('Expanded-01 best EMA',phase['parentCheckpointSHA256'])]:
        equal(expected in training_doc,True,'Documented parent '+name)
    v6=ROOT/'original/fiber-centre-compression-v6'
    config=configs['fiber-centre-compression-v6']
    equal(literal(v6/'settings.py','INITIAL_SHA'),config['initializationSHA256'],'V6 initial checkpoint')
    equal(json.loads((ROOT/'provenance/checkpoint-embedded-config.json').read_text()),config,'Embedded V6 config')
    a=json.loads((ROOT/'provenance/358_finetuned_final.json').read_text())
    b=json.loads((ROOT/'provenance/50M_MULTISCROLL_8000.json').read_text())
    equal(a['sha256'],configs['fiber-compression-multiscroll-v4']['initializationSHA256'],'V3→V4 checkpoint')
    equal(b['sha256'],configs['fiber-full-8000-v5']['initializationSHA256'],'V4→V5 checkpoint')
    for name,expected in [('V3 EMA',a['sha256']),('V4 step 8000',b['sha256']),
                          ('V5 step 874',config['initializationSHA256'])]:
        equal(expected in training_doc,True,'Documented parent '+name)
    portable=json.loads((ROOT/'provenance/released-checkpoint.json').read_text())
    equal(literal(ROOT/'original/hf-export/export.py','SOURCE_SHA256'),portable['checkpointSHA256'],'Portable/export SHA')
    hf=json.loads((ROOT/'provenance/huggingface-release.json').read_text())
    for entry in hf['files']:
        if entry['path'] in ('README.md','plans.json','dataset.json'):
            data=(ROOT/'release'/entry['path']).read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            equal(blob,entry['gitBlobId'],'HF Git blob: '+entry['path'])
    rows=json.loads((ROOT/'provenance/region-metadata.json').read_text())
    recorded={x['id']:x for x in json.loads((v6/'runs/multiscroll-centre-v6/snapshot.json').read_text())['regions']}
    for row in rows:
        equal(row['metadataSHA256'],recorded[row['id']]['filesSHA256']['metadata.json'],'Region receipt: '+row['id'])
        path=ROOT/'provenance/region-metadata-original'/(row['id']+'.json')
        equal(digest(path),row['metadataSHA256'],'Region bytes: '+row['id'])
        equal(json.loads(path.read_text()),row['metadata'],'Region values: '+row['id'])
    equal(len(rows),len(recorded),'Region coverage')
    cases=json.loads((v6/'runs/multiscroll-centre-v6/validation.json').read_text())
    equal(len(cases),136,'V6 cases')
    print(json.dumps({'checks':checks,'preservedFilesVerified':len(inventory['files']),
        'runTrackedCoreFileChecks':core,'regionMetadataRecords':len(rows),'validationCases':len(cases),
        'failures':errors,'scope':'Consistency against archived receipts and recorded HF blobs, not external attestation of training or accuracy'},indent=2))
    if errors:sys.exit(1)


if __name__ == '__main__':
    main()
