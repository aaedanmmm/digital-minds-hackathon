from pathlib import Path

from personas.storage import shard_key, completed_keys, write_record

def test_shard_key_is_stable_and_unique():
    a = shard_key("A3", "L2", "think_off", "item01")
    assert a == shard_key("A3", "L2", "think_off", "item01")
    assert a != shard_key("A3", "L2", "think_high", "item01")

def test_resume_skips_completed(tmp_path):
    prefix = str(tmp_path)
    assert completed_keys(prefix) == set()
    write_record(prefix, {"key": "A3|L2|think_off|item01", "answer": "B"})
    assert completed_keys(prefix) == {"A3|L2|think_off|item01"}

def test_records_survive_interruption(tmp_path):
    # Each record is its own file, so a killed process loses at most one.
    prefix = str(tmp_path)
    for i in range(5):
        write_record(prefix, {"key": f"k{i}", "answer": "A"})
    assert len(completed_keys(prefix)) == 5

def test_write_record_returns_the_path_it_wrote(tmp_path):
    # Callers (e.g. the runner's per-record GCS upload) need this path to
    # upload exactly the file just written, not the whole directory.
    prefix = str(tmp_path)
    path = write_record(prefix, {"key": "A3|L2|think_off|item01", "answer": "B"})
    assert Path(path).exists()
    assert Path(path).parent == tmp_path
    assert Path(path).suffix == ".json"
