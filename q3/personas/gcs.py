"""GCS sync so a preemptible Vertex worker can resume where it stopped.

`sync_up` uploads every file under a local directory. That is fine for a
one-off bulk sync, but calling it after every record turns an N-record
battery into O(N^2) uploads (re-listing and re-uploading everything written
so far, every single time) -- for the ~264-record Stage A battery that is
roughly 35,000 uploads instead of 264. `upload_file` uploads exactly the one
blob that changed, in O(1), so the runner calls it once per record and the
whole run costs O(N) uploads total. The local storage layer already
guarantees a preempted worker loses at most the one record it was mid-write
on (see personas/storage.py); calling `upload_file` right after each
`write_record` extends that same guarantee to GCS -- a preemption never
loses more than that one in-flight record remotely either.
"""
from pathlib import Path


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected a gs:// URI, got {uri}")
    body = uri[len("gs://"):]
    bucket, _, prefix = body.partition("/")
    return bucket, prefix


def _client():
    from google.cloud import storage
    return storage.Client()


def _blob_name(prefix: str, filename: str) -> str:
    return f"{prefix}/{filename}" if prefix else filename


def sync_down(gcs_prefix: str, local_dir: str) -> None:
    """Pull every blob under gcs_prefix into local_dir. Called once, before
    the run starts, so a resumed worker sees records a previous (preempted)
    worker already pushed."""
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = _client().bucket(bucket_name)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name[len(prefix):].lstrip("/")
        if name:
            blob.download_to_filename(str(Path(local_dir) / name))


def sync_up(local_dir: str, gcs_prefix: str) -> None:
    """Upload every file in local_dir. O(files in local_dir) per call --
    appropriate for a bulk/final flush, NOT for calling after every record
    (use `upload_file` for that; see module docstring)."""
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = _client().bucket(bucket_name)
    for path in Path(local_dir).glob("*"):
        if path.is_file():
            bucket.blob(_blob_name(prefix, path.name)).upload_from_filename(str(path))


def upload_file(local_path: str, gcs_prefix: str) -> None:
    """Upload exactly one local file to gcs_prefix. O(1) -- this is what the
    runner calls right after each write_record, so an N-record battery costs
    O(N) total uploads instead of the O(N^2) that calling sync_up after
    every write would cost."""
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    bucket = _client().bucket(bucket_name)
    path = Path(local_path)
    bucket.blob(_blob_name(prefix, path.name)).upload_from_filename(str(path))
