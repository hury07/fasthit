
import os


def ffindex_to_record(name):
    def build_map(l): return (l[0], l[1:])
    # index file < 100MB
    assert os.path.getsize(f"{name}.ffindex") <  100 * (1024 ** 2)
    with open(f"{name}.ffindex") as f:
        index_dbs = dict([build_map(l.rstrip('\n').split('\t')) for l in f.readlines()])
    # index file < 160GB
    assert os.path.getsize(f"{name}.ffdata") < 100 * (1024 ** 3)
    db_records = {}
    with open(f"{name}.ffdata") as f:
        for db_name, val in index_dbs.items():
            f.seek(int(val[0]), 0)
            db = f.read(int(val[1]))
            db = db.rstrip('\x00\n')
            records = [l.rstrip('\n').split(" ")[-1] for l in db.split('\n')]
            db_records[db_name] = records
    return db_records
