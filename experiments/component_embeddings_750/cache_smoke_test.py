"""Проверка восстановления настоящего кэша без загрузки модели."""
from pathlib import Path
from tempfile import TemporaryDirectory
from data_io import load_bundle
from embedding_cache import restore_cache
from embeddings import extract_embeddings

bundle=load_bundle()
with TemporaryDirectory() as temp:
    restore_cache(bundle,temp)
    result=extract_embeddings(bundle,output_dir=Path(temp))
    assert result['context'].shape==(16868,4096)
    assert result['answer_cls'].shape==(750,4096)
    print('OK: restored exact cache; model not loaded')
