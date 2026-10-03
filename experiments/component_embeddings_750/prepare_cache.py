"""Создать части точного кэша из пользовательского архива, не исполняя его код."""
import argparse
import json
import zipfile
from pathlib import Path
from data_io import ROOT, load_bundle
from embeddings import make_jobs, MODEL_ID, MODEL_REVISION
from embedding_cache import sha

def prepare(archive):
    out=ROOT/'artifacts_received'
    out.mkdir(exist_ok=True)
    names=['embeddings.npz','embeddings_manifest.json','embedding_trace.jsonl']
    with zipfile.ZipFile(archive) as z:
        for name in names:
            (out/name).write_bytes(z.read(name))
    meta=json.loads((out/names[1]).read_text(encoding='utf-8'))
    bundle=load_bundle()
    assert meta['data_signature']==bundle.signature
    assert meta['model_id']==MODEL_ID and meta['revision']==MODEL_REVISION
    assert meta['weight_quantization']=='int8'
    jobs,_=make_jobs(bundle)
    trace=[json.loads(s) for s in (out/names[2]).read_text(encoding='utf-8').splitlines()]
    actual={(r['kind'],str(r['key'])):r['text'] for r in trace}
    assert len(actual)==len(jobs)==len(trace)
    assert all(actual[(j.kind,str(j.key))]==j.text for j in jobs)
    directory=ROOT/'embedding_cache'
    directory.mkdir(exist_ok=True)
    cache=out/'cache.zip'
    with zipfile.ZipFile(cache,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for name in names:
            z.write(out/name,name)
    parts=[]
    with cache.open('rb') as stream:
        for index,block in enumerate(iter(lambda:stream.read(48*2**20),b''),1):
            path=directory/f'cache.part{index:02d}'
            path.write_bytes(block)
            parts.append(dict(name=path.name,sha256=sha(path),bytes=path.stat().st_size))
    spec=dict(fingerprint=meta['fingerprint'],archive_sha256=sha(cache),
              files={name:sha(out/name) for name in names},parts=parts,
              source='User supplied component_750_results_unverified.zip; exact bytes preserved')
    (directory/'manifest.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    print('OK exact prompts:',len(jobs),'cache parts:',len(parts))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('archive')
    prepare(parser.parse_args().archive)
