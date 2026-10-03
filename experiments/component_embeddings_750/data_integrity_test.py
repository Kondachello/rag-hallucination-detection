"""Проверка данных после преобразования переносов и из реального индекса Git."""
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from data_io import DATA, ROOT, load_bundle

def check():
    expected = load_bundle()
    names = ['manifest.json', *json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))['files']]
    with TemporaryDirectory() as temp:
        target = Path(temp)
        for ending in (b'\n', b'\r\n'):
            for name in names:
                target.joinpath(name).write_bytes(DATA.joinpath(name).read_bytes().replace(b'\r\n', b'\n').replace(b'\n', ending))
            actual = load_bundle(target)
            assert actual.signature == expected.signature and actual.ids == expected.ids
            assert actual.components == expected.components
        for name in names:
            relative = f'experiments/component_embeddings_750/data/{name}'
            raw = subprocess.check_output(['git', 'show', f':{relative}'], cwd=ROOT)
            target.joinpath(name).write_bytes(raw)
        actual = load_bundle(target)
        assert actual.signature == expected.signature and actual.components == expected.components
        path = target/'components.no_gold.jsonl'
        path.write_bytes(path.read_bytes().replace(b'not_verified', b'unknown', 1))
        try:
            load_bundle(target)
        except ValueError as error:
            assert 'components.no_gold.jsonl' in str(error)
        else:
            raise AssertionError('Повреждение содержимого осталось незамеченным')
    print('OK: LF, CRLF, реальные Git-файлы, 750 ответов; подмена содержимого отклонена.')

if __name__ == '__main__':
    check()
