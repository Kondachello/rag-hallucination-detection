"""Проверка расширенного набора и входы без целевых меток."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
TYPES = ('entity', 'relation', 'claim')
STATUSES = {'entity': ('grounded', 'ungrounded'),
            'relation': ('entailed', 'unsupported', 'contradicted', 'unknown', 'not_verified'),
            'claim': ('entailed', 'unsupported', 'contradicted', 'unknown')}
# Только доступные признаки. Неизвестные оценки отношений в признаки не входят.
FEATURES = ('entity_log_count', 'entity_ungrounded_rate', 'relation_log_count',
            'relation_missing', 'claim_log_count', 'claim_missing',
            'claim_unsupported_rate', 'claim_contradicted_rate', 'claim_unknown_rate')

def component_key(row):
    return f"{int(row['source_id'])}:{row['component_id']}"

def digest(path):
    """Хеш текстовых входов с одинаковыми переносами строк на Windows/Linux."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()

@dataclass
class Bundle:
    inputs: list
    components: list
    labels: pd.DataFrame
    confirmation: pd.DataFrame
    splits: pd.DataFrame
    ids: list
    by_id: dict
    by_component: dict
    signature: str

    @property
    def y(self):
        return self.labels.set_index('source_id').loc[self.ids, 'hallucination'].to_numpy(int)

    @property
    def q(self):
        return self.confirmation.set_index('source_id').loc[self.ids, list(FEATURES)].to_numpy(np.float32)

    @property
    def groups(self):
        return self.splits.set_index('source_id').loc[self.ids, 'group'].to_numpy()

    @property
    def partitions(self):
        return self.splits.set_index('source_id').loc[self.ids, 'partition'].to_numpy()

def load_bundle(data_dir=DATA):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('file_hash_format') != 'sha256-text-lf-v1':
        raise ValueError('Устаревший манифест данных: обновите пакет из GitHub.')
    for name, expected in manifest['files'].items():
        if digest(data_dir/name) != expected:
            raise ValueError(f'Изменился входной файл: {name}')
    read = lambda name: [json.loads(s) for s in (data_dir/name).read_text(encoding='utf-8').splitlines() if s]
    inputs, components = read('inputs.no_gold.jsonl'), read('components.no_gold.jsonl')
    labels = pd.read_csv(data_dir/'labels.csv')
    confirmation = pd.read_csv(data_dir/'confirmation_features.no_gold.csv')
    splits = pd.read_csv(data_dir/'splits.csv')
    ids = [int(row['source_id']) for row in inputs]
    if len(set(ids)) != len(ids):
        raise ValueError('Повторный source_id')
    by_id = {int(row['source_id']): row for row in inputs}
    by_component = {}
    for row in components:
        key = component_key(row)
        if key in by_component or int(row['source_id']) not in by_id:
            raise ValueError('Неверная связь компонентов')
        if row['confirmation'] not in STATUSES[row['component_type']] or not row['embedding_text'].strip():
            raise ValueError('Неверный компонент')
        by_component[key] = row
    for table in (labels, confirmation, splits):
        if table.source_id.duplicated().any() or set(table.source_id) != set(ids):
            raise ValueError('Таблицы относятся к разным ответам')
    if splits.groupby('group').partition.nunique().max() != 1:
        raise ValueError('Одна группа попала в обучение и отложенную проверку')
    bundle = Bundle(inputs, components, labels, confirmation, splits, ids,
                    by_id, by_component, manifest['dataset_signature'])
    if not np.isfinite(bundle.q).all() or set(bundle.y) != {0, 1}:
        raise ValueError('Некорректные признаки или метки')
    return bundle
