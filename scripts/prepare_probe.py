"""Generate the fixed seven-source, 350-question probe from a local original test split."""
import argparse
import hashlib
import json
import os
from pathlib import Path

SOURCES = ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']
TOTALS = [3610,11313,14267,7405,12576,2417,125]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(args.test_parquet)
    rows = table.to_pylist()
    assert len(rows)==51713
    rng = np.random.RandomState(0)
    indices = []
    for source, total in zip(SOURCES,TOTALS):
        group = [i for i,row in enumerate(rows) if row['data_source']==source]
        assert len(group)==total, source
        indices.extend(rng.choice(group,50,replace=False).tolist())
    rng.shuffle(indices)
    assert len(set(indices))==350
    args.output.mkdir(parents=True,exist_ok=False)
    selected = table.take(indices).to_pylist()
    questions = []
    for index,row in zip(indices,selected):
        row['env_kwargs']['probe_row_index']=index
        questions.append({'row_index':index,'data_source':row['data_source'],
                          'question':row['env_kwargs']['question'],
                          'ground_truth':row['env_kwargs']['ground_truth']['target']})
    probe={'seed':0,'method':'RandomState(0), each source 50 without replacement, then shuffled',
           'dataset':str(args.test_parquet.resolve()),'row_indices':indices,'questions':questions}
    path=args.output/'probe.json'
    path.write_text(json.dumps(probe,ensure_ascii=False,indent=2),encoding='utf-8')
    with path.open('rb') as stream:os.fsync(stream.fileno())
    pq.write_table(pa.Table.from_pylist(selected),args.output/'validation_probe.parquet')
    print('C3_PROBE_JSON='+str(path.resolve()))
    print('C3_PROBE_PARQUET='+str((args.output/'validation_probe.parquet').resolve()))
    print('C3_PROBE_SHA256='+hashlib.sha256(path.read_bytes()).hexdigest())

if __name__=='__main__':main()
