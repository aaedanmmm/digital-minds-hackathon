import pytest

import personas.gcs as gcs
from personas.gcs import parse_gcs_uri


def test_parses_bucket_and_prefix():
    assert parse_gcs_uri("gs://b/x/y") == ("b", "x/y")


def test_parses_bucket_only():
    assert parse_gcs_uri("gs://b") == ("b", "")


def test_rejects_non_gcs():
    with pytest.raises(ValueError, match="gs://"):
        parse_gcs_uri("/local/path")


# --- fakes standing in for google.cloud.storage, so these tests never need
# the real package installed or real network access ---

class FakeListedBlob:
    def __init__(self, name, content=b"downloaded"):
        self.name = name
        self._content = content

    def download_to_filename(self, path):
        from pathlib import Path
        Path(path).write_bytes(self._content)


class FakeUploadBlob:
    def __init__(self, bucket, name):
        self._bucket = bucket
        self.name = name

    def upload_from_filename(self, local_path):
        self._bucket.uploads.append((self.name, local_path))


class FakeBucket:
    def __init__(self, name, listed_blobs=None):
        self.name = name
        self.uploads = []
        self._listed_blobs = listed_blobs or []

    def blob(self, name):
        return FakeUploadBlob(self, name)

    def list_blobs(self, prefix=""):
        return [b for b in self._listed_blobs if b.name.startswith(prefix)]


class FakeClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        assert name == self._bucket.name
        return self._bucket


def test_sync_down_downloads_every_blob_under_prefix(tmp_path, monkeypatch):
    bucket = FakeBucket("b", [
        FakeListedBlob("prefix/a.json", b'{"key": "a"}'),
        FakeListedBlob("prefix/b.json", b'{"key": "b"}'),
        FakeListedBlob("other/c.json", b'{"key": "c"}'),  # not under prefix
    ])
    monkeypatch.setattr(gcs, "_client", lambda: FakeClient(bucket))

    gcs.sync_down("gs://b/prefix", str(tmp_path))

    assert (tmp_path / "a.json").read_bytes() == b'{"key": "a"}'
    assert (tmp_path / "b.json").read_bytes() == b'{"key": "b"}'
    assert not (tmp_path / "c.json").exists()


def test_sync_up_uploads_every_file_in_dir(tmp_path, monkeypatch):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    bucket = FakeBucket("b")
    monkeypatch.setattr(gcs, "_client", lambda: FakeClient(bucket))

    gcs.sync_up(str(tmp_path), "gs://b/prefix")

    uploaded_names = {name for name, _ in bucket.uploads}
    assert uploaded_names == {"prefix/a.json", "prefix/b.json"}


def test_upload_file_uploads_only_the_named_file(tmp_path, monkeypatch):
    """Defect 2 fix: unlike sync_up, upload_file touches exactly one blob,
    so calling it after every record is O(1) per call instead of O(n)."""
    target = tmp_path / "record123.json"
    target.write_text("{}")
    (tmp_path / "unrelated.json").write_text("{}")  # must NOT be uploaded
    bucket = FakeBucket("b")
    monkeypatch.setattr(gcs, "_client", lambda: FakeClient(bucket))

    gcs.upload_file(str(target), "gs://b/prefix")

    assert bucket.uploads == [("prefix/record123.json", str(target))]
