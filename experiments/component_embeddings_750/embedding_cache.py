"""Точный пользовательский кэш, разбитый на файлы меньше лимита GitHub."""
import hashlib
import json
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from embeddings import MODEL_ID, MODEL_REVISION, PROMPT_VERSION

ROOT = Path(__file__).resolve().parent

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()

def restore_cache(bundle, output_dir):
    directory = ROOT/'embedding_cache'
    spec = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    fingerprint = hashlib.sha256((bundle.signature+MODEL_ID+MODEL_REVISION+PROMPT_VERSION+'int8').encode()).hexdigest()
    if spec['fingerprint'] != fingerprint:
        raise ValueError('Сохранённый кэш не соответствует данным или кодировщику.')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if all((out/name).is_file() and sha(out/name)==value for name,value in spec['files'].items()):
        print('Используем проверенный локальный кэш эмбеддингов.')
        return
    with TemporaryDirectory(dir=out) as temp:
        archive = Path(temp)/'cache.zip'
        with archive.open('wb') as stream:
            for part in spec['parts']:
                path = directory/part['name']
                if sha(path) != part['sha256']:
                    raise ValueError('Повреждена часть кэша: '+part['name'])
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(2**20), b''):
                        stream.write(block)
        if sha(archive) != spec['archive_sha256']:
            raise ValueError('Повреждён архив кэша.')
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != set(spec['files']):
                raise ValueError('Неверный состав кэша.')
            for name, expected in spec['files'].items():
                path = Path(temp)/name
                with z.open(name) as source, path.open('wb') as target:
                    for block in iter(lambda: source.read(2**20), b''):
                        target.write(block)
                if sha(path) != expected:
                    raise ValueError('Повреждён файл кэша: '+name)
            for name in spec['files']:
                (Path(temp)/name).replace(out/name)
    print('Восстановлены готовые эмбеддинги из GitHub; загрузка 8B не нужна.')
