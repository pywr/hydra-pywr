import hashlib
import uuid

from hydra_pywr.utils import get_source_upload_appdata


def test_uploaded_file_is_described(tmp_path):
    data_uuid = str(uuid.uuid4())
    sources = tmp_path / data_uuid / 'model_sources'
    sources.mkdir(parents=True)
    f = sources / '021026130630_BV11_SDB.json'
    f.write_bytes(b'{}')

    source = get_source_upload_appdata(str(f))

    assert source['data_uuid'] == data_uuid
    assert source['stored_name'] == '021026130630_BV11_SDB.json'
    assert source['original_name'] == 'BV11_SDB.json'
    assert source['size'] == 2
    assert source['sha256'] == hashlib.sha256(b'{}').hexdigest()


def test_file_not_uploaded_through_hwi_is_ignored(tmp_path):
    f = tmp_path / 'model.json'
    f.write_text('{}')
    assert get_source_upload_appdata(str(f)) is None
    assert get_source_upload_appdata(None) is None


def test_non_uuid_project_folder_is_ignored(tmp_path):
    sources = tmp_path / 'not-a-uuid' / 'model_sources'
    sources.mkdir(parents=True)
    f = sources / 'x.json'
    f.write_text('{}')
    assert get_source_upload_appdata(str(f)) is None
