"""Download and verify the SAM2.1 release weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

from sghaquant.runtime import MODELS, ROOT, SEEDS

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def install(name, record, manifest, destination, source=None):
    if all((destination / f['path']).is_file() and sha256(destination / f['path']) == f['sha256']
           for f in record['files']):
        print(f'{name}: verified', flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=destination) as temp:
        temp = Path(temp)
        archive = temp / name
        if source:
            shutil.copyfile(source / name, archive)
        else:
            url = f'https://github.com/{manifest["repository"]}/releases/download/{manifest["release"]}/{name}'
            print(f'{name}: downloading {record["bytes"] / 1e6:.1f} MB', flush=True)
            with urlopen(url, timeout=120) as response, archive.open('wb') as target:
                shutil.copyfileobj(response, target, 8 * 1024 * 1024)
        if archive.stat().st_size != record['bytes'] or sha256(archive) != record['sha256']:
            raise RuntimeError(f'Archive checksum mismatch: {name}')
        expected = {f['path']: f for f in record['files']}
        extracted = temp / 'extracted'
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(expected):
                raise RuntimeError(f'Unexpected files in {name}')
            for member in members:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                    raise RuntimeError(f'Invalid archive member: {member.name}')
                target = extracted / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as stream, target.open('wb') as output:
                    shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
                if target.stat().st_size != expected[member.name]['bytes'] or sha256(target) != expected[member.name]['sha256']:
                    raise RuntimeError(f'File checksum mismatch: {member.name}')
        for relative in expected:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(extracted / relative, target)
    print(f'{name}: installed', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=[*MODELS, 'all'], default='tiny')
    p.add_argument('--precision', choices=['w4a4', 'w8a8', 'both'], default='both')
    p.add_argument('--seeds', default=str(SEEDS[0]), help='all or comma-separated calibration seeds')
    p.add_argument('--output', type=Path, default=ROOT / 'weights')
    p.add_argument('--source-directory', type=Path, help='install previously downloaded release archives')
    args = p.parse_args()
    seeds = SEEDS if args.seeds == 'all' else [int(s) for s in args.seeds.split(',')]
    if any(s not in SEEDS for s in seeds):
        p.error(f'Available seeds: {SEEDS}')
    manifest = json.loads((ROOT / 'weights.json').read_text())
    models = list(MODELS) if args.model == 'all' else [args.model]
    kinds = ['reference']
    if args.precision in ('w4a4', 'both'):
        kinds += [f'w4a4-{s}' for s in seeds]
    if args.precision in ('w8a8', 'both'):
        kinds += ['w8a8']
    for name, record in manifest['assets'].items():
        if record['model'] in models and record['kind'] in kinds:
            install(name, record, manifest, args.output.resolve(), args.source_directory)

if __name__ == '__main__':
    main()
