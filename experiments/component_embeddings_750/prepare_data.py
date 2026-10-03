"""Однократная сборка из пользовательского архива, без выполнения его кода."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from data_io import DATA, FEATURES, load_bundle

def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def prepare(archive, pilot_labels):
    with tarfile.open(archive) as tar:
        read = lambda name: tar.extractfile('ec750/owner-private/'+name).read()
        labels_raw = pd.read_csv(io.BytesIO(read('inputs/annotated_answers.csv')))
        feedback = {}
        relations = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if '/run/feedback/' in member.name:
                row = json.load(tar.extractfile(member))
                if row['entity_status'] == row['claim_status'] == 'ok':
                    key = (int(row['source_id']), row['answer_sha256'])
                    if key in feedback and feedback[key] != row:
                        raise ValueError('Неоднозначные успешные оценки')
                    feedback[key] = row
            if '/relation-extraction/sources/' in member.name:
                row = json.load(tar.extractfile(member))
                relations[int(row['source_id'])] = row
    old_ids = set(pd.read_csv(pilot_labels).source_id.astype(int))
    inputs, components, labels, features = [], [], [], []
    for row in labels_raw.to_dict('records'):
        sid = int(row['id'].rsplit('_', 1)[-1])
        answer = row['generated_response']
        prompt = row['prompt']
        match = re.search(r'Briefly answer the following question:\s*(.*?)\s*Bear in mind that your response should be strictly based on the following three passages:\s*(.*?)\s*In case the passages do not contain', prompt, re.S)
        if match is None:
            raise ValueError(f'Не распознан исходный запрос: {sid}')
        query, context = match.group(1).strip(), match.group(2).strip()
        fb = feedback[(sid, sha(answer))]
        rel = relations[sid]
        if any(rel[k] != sha(value) for k, value in [('answer_sha256', answer), ('query_sha256', query), ('context_sha256', context)]):
            raise ValueError(f'Тройкам соответствует другой текст: {sid}')
        inputs.append(dict(source_id=sid, response_id=row['id'], answer=answer,
                           query=query, context=context))
        labels.append(dict(source_id=sid, response_id=row['id'], hallucination=int(row['hallucination']), annotation_model=row['annotation_model']))
        count = Counter()
        def add(kind, text, status, start=None, end=None):
            if not text.strip():
                raise ValueError('Пустой компонент')
            count[kind] += 1
            components.append(dict(source_id=sid, response_id=row['id'],
                component_id=f'{kind}_{count[kind]}', component_type=kind,
                embedding_text=text, confirmation=status, answer_start=start, answer_end=end))
        for entity in fb['entities']:
            start, end = entity.get('start'), entity.get('end')
            if start is None or end is None or answer[start:end].casefold() != entity['name'].casefold():
                start = end = None
            add('entity', entity['name'], 'grounded' if entity['grounded'] else 'ungrounded', start, end)
        for triple in rel['relations']['answer']:
            if len(triple) != 3:
                raise ValueError('Неверная тройка')
            add('relation', f'subject: {triple[0]}; predicate: {triple[1]}; object: {triple[2]}', 'not_verified')
        for claim in fb['claims']:
            add('claim', claim['text'], claim['verdict'])
        ec = Counter('grounded' if e['grounded'] else 'ungrounded' for e in fb['entities'])
        cc = Counter(c['verdict'] for c in fb['claims'])
        features.append(dict(source_id=sid, response_id=row['id'],
            entity_log_count=np.log1p(count['entity']), entity_ungrounded_rate=ec['ungrounded']/max(1,count['entity']),
            relation_log_count=np.log1p(count['relation']), relation_missing=int(count['relation']==0),
            claim_log_count=np.log1p(count['claim']), claim_missing=int(count['claim']==0),
            claim_unsupported_rate=cc['unsupported']/max(1,count['claim']),
            claim_contradicted_rate=cc['contradicted']/max(1,count['claim']),
            claim_unknown_rate=cc['unknown']/max(1,count['claim'])))
    # Общая группа для одинакового контекста/вопроса или одинакового ответа.
    parent = list(range(len(inputs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(inputs):
        norm = lambda s: ' '.join(s.casefold().split())
        for key in ('input:'+sha(norm(row['query'])+'\n'+norm(row['context'])),
                    'answer:'+sha(norm(row['answer']))):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = np.array([find(i) for i in range(len(inputs))])
    y = np.array([row['hallucination'] for row in labels])
    old_groups = {groups[i] for i, row in enumerate(inputs) if row['source_id'] in old_ids}
    eligible = np.array([i for i in range(len(inputs)) if groups[i] not in old_groups])
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=20261003)
    _, test_local = next(splitter.split(eligible, y[eligible], groups[eligible]))
    test_groups = set(groups[eligible[test_local]])
    splits = [dict(source_id=row['source_id'], group=int(groups[i]),
        partition='holdout' if groups[i] in test_groups else 'development',
        seen_in_pilot=int(row['source_id'] in old_ids)) for i,row in enumerate(inputs)]
    DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in [('inputs.no_gold.jsonl', inputs), ('components.no_gold.jsonl', components)]:
        (DATA/name).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    for name, rows in [('labels.csv',labels), ('confirmation_features.no_gold.csv', features), ('splits.csv',splits)]:
        pd.DataFrame(rows).to_csv(DATA/name,index=False)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.iterdir()) if p.name!='manifest.json'}
    manifest = dict(files=files, dataset_signature=sha(json.dumps(files,sort_keys=True)),
        archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(), n_answers=len(inputs),
        component_counts=dict(Counter(c['component_type'] for c in components)),
        class_counts=dict(Counter(str(v) for v in y)), n_groups=len(set(groups)),
        n_pilot_overlap=sum(r['seen_in_pilot'] for r in splits),
        partition_counts=dict(Counter(r['partition'] for r in splits)),
        relation_confirmation='Unavailable; all relations are not_verified; excluded from confirmation features',
        label_source='openai/gpt-4o; original answers only; revised answers excluded',
        features=list(FEATURES))
    (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    load_bundle()
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('archive',type=Path)
    parser.add_argument('--pilot-labels',type=Path,required=True)
    args=parser.parse_args(); prepare(args.archive,args.pilot_labels)
