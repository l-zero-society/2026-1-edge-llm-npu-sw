#!/usr/bin/env python3
"""Atomically reuse completed sweeps ONLY when full run identities match."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--destination',type=Path,default=Path('diagnostics'))
    args=parser.parse_args()
    source_provenance=args.source/'diagnosis_provenance.json'
    dest_provenance=args.destination/'diagnosis_provenance.json'
    if json.loads(source_provenance.read_text())['identity']!=json.loads(dest_provenance.read_text())['identity']:
        raise ValueError('Refusing checkpoint import: provenance identities differ')
    audit_file=args.destination/'checkpoint_imports.json'
    audit=json.loads(audit_file.read_text()) if audit_file.exists() else []
    imported=0
    for source in sorted((args.source/'sweeps').glob('*.json')):
        data=json.loads(source.read_text())
        if len(data['attribution'])!=2 or {r['split'] for r in data['attribution']}!={'calibration','validation'}:
            raise ValueError('incomplete attribution in sweep checkpoint')
        if len(data['input_sweep'])!=14 or len(data['output_sweep'])!=12:
            raise ValueError('incomplete sweep checkpoint')
        destination=args.destination/'sweeps'/source.name
        if destination.exists():continue  # never overwrite an independently completed checkpoint
        destination.parent.mkdir(exist_ok=True)
        temporary=destination.with_suffix('.import.tmp')
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        audit.append(dict(source=str(source),destination=str(destination),sha256=digest(source),
                          source_provenance_sha256=digest(source_provenance),exact_identity_equal=True,
                          time=time.time(),command=sys.argv,reason='completed identical-identity sweep'))
        imported+=1
    temporary=audit_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(audit,indent=2)+'\n');temporary.replace(audit_file)
    print(f'Imported {imported} completed sweeps; exact identities match')


if __name__=='__main__':main()
