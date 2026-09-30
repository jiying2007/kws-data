#!/usr/bin/env python3
"""Verify retained host evidence using Python's standard library; never run models."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

DEFAULT = Path(__file__).resolve().parents[1] / 'reports/host-resource-profiles-20260930'

REQUIRED_FILES = frozenset(('README.md', 'cfsmn/RESULTS.md', 'cfsmn/profile-corrected.json', 'cfsmn/profile.json', 'cfsmn/profile.py', 'cfsmn/run.log', 'frozen-c/PROFILE_NOTES.md', 'frozen-c/build-receipt.json', 'frozen-c/cpu-io-run-1.json', 'frozen-c/cpu-io-run-2.json', 'frozen-c/cpu-io-run-3.json', 'frozen-c/generated/kws_build_config.h', 'frozen-c/generated/kws_parameter_limits.h', 'frozen-c/host-info.json', 'frozen-c/preliminary-board-bench.json', 'frozen-c/profile_cpu_io.c', 'lightweight-ab/A-console.json', 'lightweight-ab/A-profile.json', 'lightweight-ab/A-timings.npz', 'lightweight-ab/B-console.json', 'lightweight-ab/B-profile.json', 'lightweight-ab/B-timings.npz', 'lightweight-ab/REPORT.md', 'lightweight-ab/resource_profile.py', 'sherpa-fp32/README.md', 'sherpa-fp32/measurement.json', 'sherpa-fp32/profile.py', 'sherpa-fp32/raw-timings.json', 'sherpa-fp32/stderr.log', 'sherpa-fp32/stdout.log', 'sherpa-int8/PROTOCOL.md', 'sherpa-int8/README.md', 'sherpa-int8/comparison.json', 'sherpa-int8/config.json', 'sherpa-int8/evidence/downloads.json', 'sherpa-int8/evidence/model-README.md', 'sherpa-int8/evidence/pinned-files.json', 'sherpa-int8/evidence/runtime-LICENSE', 'sherpa-int8/keywords.txt', 'sherpa-int8/manifest.json', 'sherpa-int8/resource-profile/measurement.json', 'sherpa-int8/resource-profile/profile.py', 'sherpa-int8/resource-profile/raw-timings.json', 'sherpa-int8/resource-profile/stderr.log', 'sherpa-int8/resource-profile/stdout.log', 'sherpa-int8/results/raw-events.jsonl', 'sherpa-int8/results/readback.json', 'sherpa-int8/results/run-provenance.json', 'sherpa-int8/results/stderr.log', 'sherpa-int8/results/stdout.jsonl', 'sherpa-int8/run.py', 'sherpa-int8/validate.py'))

def close(actual, expected, atol=1e-9):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise ValueError(f'Invalid numeric evidence: {actual!r}')
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=atol):
        raise ValueError(f'Arithmetic mismatch: {actual} != {expected}')

def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid timing denominator')
    return value

def load(root, name):
    return json.loads((root / name).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_ns(path, member):
    # Deliberately support only this archived format, never pickle/object dtype.
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ['cpu_ns.npy', 'wall_ns.npy']:
            raise ValueError('Unexpected timing ZIP members')
        data = archive.read(member + '.npy')
    if data[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError('Unsupported timing NPY header')
    n = int.from_bytes(data[8:10], 'little')
    header = ast.literal_eval(data[10:10+n].decode('ascii'))
    if header != {'descr':'<i8', 'fortran_order':False, 'shape':(3000,)}:
        raise ValueError('Unexpected raw timing shape/dtype')
    values = struct.unpack('<3000q', data[10+n:])
    if any(v < 0 for v in values):
        raise ValueError('Negative raw time')
    return values

def percentile(values, fraction):
    values=sorted(values);position=(len(values)-1)*fraction
    low=math.floor(position);high=math.ceil(position)
    return values[low]+(values[high]-values[low])*(position-low)

def check_percentiles(summary, values, divisor, suffix):
    for q in (50,95,99):
        close(summary[f'p{q}_{suffix}'],percentile(values,q/100)/divisor)


def verify(root=DEFAULT):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Archive root symlink rejected')
    root = root.resolve()
    manifest = load(root, 'manifest.json')
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or manifest.get('purpose') != 'host-resource-evidence-not-target-qualification':
        raise ValueError('Invalid manifest schema/purpose')
    seen = set()
    for item in manifest['files']:
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or item['path'] in seen:
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(item['path'])
        p = root / rel
        if type(item.get('bytes')) is not int or item['bytes'] < 0 or not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']):
            raise ValueError('Invalid manifest bytes/hash schema')
        if not p.resolve().is_relative_to(root) or any((root / Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):
            raise ValueError('Archive symlink/containment violation')
        if p.is_symlink() or not p.is_file() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError(f'File identity mismatch: {rel}')
        if p.suffix != '.npz':
            p.read_text(encoding='utf-8')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} - {'manifest.json'}
    if seen != REQUIRED_FILES:
        raise ValueError('Missing or unexpected mandatory inventory')
    if seen != actual:
        raise ValueError('Manifest does not cover exact pack files')
    build = load(root, 'frozen-c/build-receipt.json')
    if sha(root/'frozen-c/profile_cpu_io.c') != build['cpu_harness']['source_sha256'] or any(sha(root/'frozen-c/generated'/name) != h for name,h in build['generated'].items()):
        raise ValueError('C build source/generated identity mismatch')
    for i in range(1, 4):
        d = load(root, f'frozen-c/cpu-io-run-{i}.json')
        exposure = positive(d['measured_audio_seconds'])
        close(d['audio_seconds_per_repeat'], build['workload']['frames']/16000)
        close(exposure, d['audio_seconds_per_repeat'] * d['measured_repeats'])
        close(exposure, d['measured_blocks'] * .020)
        close(d['process_cpu_rtf'], d['process_cpu_us'] / 1e6 / exposure)
        close(d['wall_rtf'], d['summed_block_wall_us'] / 1e6 / exposure)
    d = load(root, 'cfsmn/profile-corrected.json')
    original = load(root, 'cfsmn/profile.json')
    corrected = json.loads(json.dumps(original))
    corrected['state_observed_max']['wave_remained']['max_backing_storage_bytes'] = None
    corrected['post_measurement_correction'] = d['post_measurement_correction']
    if corrected != d:
        raise ValueError('Unexpected changes in annotated cFSMN profile')
    if sha(root / 'cfsmn/profile.py') != d['profile_sha256']:
        raise ValueError('cFSMN measurement script drift')
    if len(d['passes']) != 3:
        raise ValueError('cFSMN needs three measured passes')
    for run in d['passes']:
        exposure = positive(run['audio_s'])
        close(exposure, sum(x['audio_s'] for x in run['recordings']))
        close(run['pipeline_cpu_s'], sum(x['cpu_s'] for x in run['recordings']))
        close(run['pipeline_wall_s'], sum(x['wall_s'] for x in run['recordings']))
        close(run['pipeline_cpu_rtf'], run['pipeline_cpu_s']/exposure)
        close(run['pipeline_wall_rtf'], run['pipeline_wall_s']/exposure)
        if len(run['recordings']) != 42 or len({r['recording'] for r in run['recordings']}) != 42 or not all(x['event_and_logits_identical'] is True for x in run['recordings']):
            raise ValueError('cFSMN workload/parity incomplete')
        for c in run['components'].values():
            close(c['cpu_rtf'], c['cpu_s']/exposure)
            close(c['wall_rtf'], c['wall_s']/exposure)
    for prefix in ('sherpa-fp32/', 'sherpa-int8/resource-profile/'):
        d = load(root, prefix+'measurement.json')
        raw = load(root, prefix+'raw-timings.json')
        if sha(root/(prefix+'profile.py')) != d['script_sha256']:
            raise ValueError('sherpa measurement script drift')
        close(d['audio_preload']['audio_seconds'], d['audio_preload']['total_pcm_frames']/16000)
        if len(d['runs']) != d['protocol']['measured_passes']:
            raise ValueError('sherpa pass count mismatch')
        for run, timings in zip(d['runs'], raw, strict=True):
            close(run['audio_s'], d['audio_preload']['audio_seconds'])
            exposure = positive(run['audio_s'])
            close(run['cpu_rtf'], run['cpu_s']/exposure)
            close(run['wall_rtf'], run['wall_s']/exposure)
            close(run['feed_plus_ready_decode']['count'], len(timings['feed_ns']))
            close(run['feed_plus_ready_decode']['max_ms'], max(timings['feed_ns'])/1e6)
            check_percentiles(run['feed_plus_ready_decode'],timings['feed_ns'],1e6,'ms')
            if run['clips'] != 42 or len(timings['eof_ns']) != 42:
                raise ValueError('sherpa workload mismatch')
        check_percentiles(d['steady']['feed_plus_ready_decode'], [t for r in raw for t in r['feed_ns']], 1e6, 'ms')
        for k in ('audio_s', 'cpu_s', 'wall_s'):
            close(d['steady'][k], sum(x[k] for x in d['runs']))
        close(d['steady']['cpu_rtf'], d['steady']['cpu_s']/positive(d['steady']['audio_s']))
    provenance = load(root, 'sherpa-int8/results/run-provenance.json')
    for file,key in [('run.py','script_sha256'),('config.json','config_sha256'),('keywords.txt','keywords_sha256'),('results/readback.json','readback_sha256'),('evidence/downloads.json','downloads_sha256')]:
        if sha(root/'sherpa-int8'/file) != provenance[key]:
            raise ValueError('INT8 provenance mismatch')
    comparison = load(root, 'sherpa-int8/comparison.json')
    if sha(root/'sherpa-fp32/measurement.json') != comparison['fp32_measurement_sha256'] or sha(root/'sherpa-int8/resource-profile/measurement.json') != comparison['int8_measurement_sha256']:
        raise ValueError('Sequential comparison identity mismatch')
    fpraw = load(root, 'sherpa-fp32/raw-timings.json')
    intraw = load(root, 'sherpa-int8/resource-profile/raw-timings.json')
    recordings = load(root, 'sherpa-int8/results/readback.json')['recordings']
    if len(fpraw) != 3 or len(intraw) != 3 or len(recordings) != 42:
        raise ValueError('Incomplete INT8 comparison')
    events = [dict(recording=r['recording'], phrase=e['phrase'], after_samples=e['available_audio_samples'], eof=e['eof_flush']) for r in recordings for e in r['events']]
    for fp, quant in zip(fpraw, intraw, strict=True):
        if fp['events'] != quant['events'] or quant['events'] != events:
            raise ValueError('INT8 event parity mismatch')
    close(sum(r['frames']/16000 for r in recordings), 78.56)
    for name in ('A','B'):
        d = load(root, f'lightweight-ab/{name}-profile.json')
        timings = root/f'lightweight-ab/{name}-timings.npz'
        if sha(timings) != d['timings_sha256'] or sha(root/'lightweight-ab/resource_profile.py') != d['profile_script_sha256']:
            raise ValueError('Lightweight timings/script drift')
        protocol = d['protocol']
        close(protocol['measured_frames'], positive(protocol['frames_per_repeat'])*protocol['repeats'])
        cpu = read_ns(timings, 'cpu_ns'); wall = read_ns(timings, 'wall_ns')
        close(d['process_cpu_per_frame']['mean_us'], sum(cpu)/len(cpu)/1000)
        close(d['wall_per_frame']['mean_us'], sum(wall)/len(wall)/1000)
        check_percentiles(d['process_cpu_per_frame'],cpu,1000,'us')
        check_percentiles(d['wall_per_frame'],wall,1000,'us')
        close(d['host_one_core_equivalent_at50fps_percent_from_cpu_mean'], sum(cpu)/len(cpu)/1e9/(positive(protocol['frame_hop_ms'])/1000)*100)
    return {'verified_files':len(seen), 'verified_bytes':sum(x['bytes'] for x in manifest['files']), 'scope':'retained host evidence only, no inference or SSC305 validation'}

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT)
    print(json.dumps(verify(parser.parse_args().root),sort_keys=True))
