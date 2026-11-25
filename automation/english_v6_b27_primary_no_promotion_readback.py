"""Read only the one original B27 PRIMARY through its captured import cohort.

The first independent continuation stopped because the original reader loaded
an evaluator from a second extraction while its cached profile still belonged
to the first root. This adapter authenticates and extracts the original source
before importing any project module. A restrictive finder keeps every normal
project import in that one captured root even when the unchanged evaluator
adds another extraction to sys.path. The original reader, exact path/hash
checks, proof ContextVars, offline child guards, and verification function are
not replaced, patched, cloned, or weakened.

Only the exact already-completed B27 NO_PROMOTION can be certified here. The
returned wrapper explicitly identifies this adapter as well as the untouched
original reader; it cannot claim that the original standalone invocation
succeeded, cannot certify PASS, and grants no scientific execution permission.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import importlib.abc
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import tempfile
from unittest import mock
import zipfile

REPORT_SCHEMA = 'english-v6-b27-primary-rooted-no-promotion-readback-v1'
CHECK_SCHEMA = 'english-v6-b27-original-captured-import-check-v1'
READER_NAME = 'english_v6_b27_quality_readback'
READER_SHA = '87bc31725380cbf70d70d6b46929a724dc6bff656d421c5ae63b54c0724e31aa'
PRIMARY_NAME = 'english_v6_b27_primary_eval'
PRIMARY_SHA = '0810522c031da4e9e8d4d3eadcfa2cecd0365277c8bec557448c8b8315e97d49'
RUN_ID = 45627775991
RUN_HEAD = 'dce5b169163039091afcc5ab62beed3211d37552'
PREPARED_PARENT = '8981de96d00e4ac6d6a1b2e65aea6a7023d1d3dc'
CANDIDATE_PT = 'dd0d4bc08be1162886ab462dfb3ad7f4503150a57ee13272d01f41e80ca1069f'
ARTIFACTS = {
    'result': (15036520814, 6118732, 'c65d5e4f3f1c9046258d3677cac17b6406981fd6bfd8dc512a6e846158554c62'),
    'producer': (15036515900, 34622369, '082ec32ca703ca5a9c1ca5e9b1a7e70aefd6cbab47f855a4c96250e1d7736c64'),
    'consumed': (15032429755, 382, 'be6d9435a9662ed06c3e3e5d12e98d07272ed79cd91cd8e08d80dd1421cde0e4'),
}
ORIGINAL_HASHES = {
    'evaluator_sha256': PRIMARY_SHA,
    'inputs_sha256': '66b39d8a0372d658e83a97f66db3edc421b954b242ecc9a40ca560951d904bda',
    'producer_manifest_sha256': '1a174d05f22254009f1d8023cf5e2366edb44941e6825d7bf66c7a1a1b678ebe',
    'original_result_sha256': '05cebb6b9074677cff7f701fb5fbc455f20ceba7de1255935d0753c1db02fb27',
    'external_proof_binding_sha256': '6c6334a1d8d6452e186c41fe472eca6e072a8af42a02a81d9aac68b30e7d9a28',
}
FORBIDDEN = frozenset(('numpy', 'torch', 'torchaudio', 'onnxruntime', 'soundfile',
                       'transformers', 'piper', 'requests', 'httpx'))
MAX_BYTES = 128_000_000
RESEARCH_MODULES = {
    'campaign': 'research/auto-tts-voice-clone-20261102/campaign.py',
    'runner': 'research/auto-tts-voice-clone-20261102/runner.py',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def document(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    need(type(value) is dict, 'one exact JSON object')
    return value


def exact_lock(lock):
    need(type(lock) is dict and set(lock) == {'schema', 'panel', *ARTIFACTS, *ORIGINAL_HASHES}
         and lock['schema'] == 'english-v6-b27-original-quality-lock-v1'
         and lock['panel'] == 'primary', 'only the exact original B27 PRIMARY lock')
    for role, (identifier, size, digest) in ARTIFACTS.items():
        need(type(lock[role]) is dict
             and all(type(lock[role].get(key)) is int for key in ('id', 'bytes', 'run_id'))
             and lock[role] == dict(id=identifier, bytes=size, sha256=digest,
             run_id=RUN_ID, run_head=RUN_HEAD), 'the one original B27 artifact ' + role)
    need(all(lock[key] == value for key, value in ORIGINAL_HASHES.items()),
         'unchanged original B27 code, result, inputs and full external proof')
    return lock


def regular(path, *, directory=False):
    path = Path(path)
    need(path.is_absolute() and path == path.resolve()
         and all(not parent.is_symlink() for parent in (path, *path.parents)),
         'regular resolved original path')
    need(path.is_dir() if directory else path.is_file(), 'complete original file/directory exists')
    return path


def authenticated_archive(path, role):
    need(role in ARTIFACTS, 'one explicit original artifact role')
    raw = regular(path).read_bytes()
    _identifier, size, digest = ARTIFACTS[role]
    need(len(raw) == size and sha(raw) == digest, 'exact whole original artifact before any code import')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    try:
        infos = archive.infolist()
        need(len(infos) <= 8192 and len({item.filename for item in infos}) == len(infos)
             and sum(item.file_size for item in infos) <= MAX_BYTES, 'bounded unique complete original ZIP')
        for item in infos:
            name = PurePosixPath(item.filename)
            need(name.parts and not name.is_absolute() and '..' not in name.parts
                 and '\\' not in item.filename and '\0' not in item.filename
                 and not stat.S_ISLNK(item.external_attr >> 16), 'regular normalized original ZIP member')
        need(archive.testzip() is None, 'complete original ZIP CRC')
        return archive
    except BaseException:
        archive.close()
        raise


def extract_original_root(producer_path, destination):
    destination = Path(destination)
    need(destination.is_absolute() and destination == destination.resolve()
         and destination.parent.is_dir() and not destination.exists(), 'fresh exact captured-root destination')
    pins = {}
    with authenticated_archive(producer_path, 'producer') as archive:
        names = [item.filename for item in archive.infolist()
                 if not item.is_dir() and item.filename.startswith('primary_producer/')]
        need(len(names) == 479, 'all479 original PRIMARY producer files, with no pruning')
        destination.mkdir()
        for member in names:
            relative = member.removeprefix('primary_producer/')
            path = PurePosixPath(relative)
            need(path.as_posix() == relative and all(part not in ('', '.', '..') for part in relative.split('/')),
                 'normalized original producer relative path')
            body = archive.read(member)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(body)
            pins[relative] = dict(bytes=len(body), sha256=sha(body))
    need(pins['automation/' + READER_NAME + '.py']['sha256'] == READER_SHA
         and pins['automation/' + PRIMARY_NAME + '.py']['sha256'] == PRIMARY_SHA,
         'original independent verifier and evaluator remain byte-identical')
    return pins


def original_failure_preflight(directory):
    """Authenticate the original raw failure before choosing this adapter."""
    with authenticated_archive(Path(directory) / 'originals/result.zip', 'result') as archive:
        raw = archive.read('RESULT.json')
    need(sha(raw) == ORIGINAL_HASHES['original_result_sha256'], 'exact original RESULT member')
    result = document(raw)
    need(result['schema'] == 'english-v6-b27-fresh-primary-four-frozen-models-v1'
         and result['panel'] == 'primary' and result['panel_gate'] == 'NO_PROMOTION'
         and result['run_id'] == str(RUN_ID) and result['run_head'] == RUN_HEAD
         and type(result['run_attempt']) is int and result['run_attempt'] == 1
         and result['prepared_parent'] == PREPARED_PARENT
         and result['model_pt_sha256']['challenger'] == CANDIDATE_PT
         and all(type(result[key]) is int and result[key] == 996
                 for key in ('actual_forwards', 'saved_array_count', 'raw_output_count'))
         and result['consumed_unknown'] == []
         and 'error' not in result and 'error_type' not in result
         and all(result[key] is False for key in ('replication_open_allowed',
             'candidate_engineering_validation_allowed', 'promotion_allowed', 'product_released', 'replay_allowed')),
         'only the complete original raw B27 failure routes to this adapter; this is not certification')
    return result


def captured_module_name(relative):
    path = PurePosixPath(relative)
    need(path.as_posix() == relative and not path.is_absolute() and '..' not in path.parts
         and path.suffix == '.py', 'one normalized captured Python source path')
    if path.parent.as_posix() == 'automation':
        parts = (path.stem,)
    elif relative in RESEARCH_MODULES.values():
        parts = (next(name for name, value in RESEARCH_MODULES.items() if value == relative),)
    else:
        parts = path.parts[:-1] if path.name == '__init__.py' else (*path.parts[:-1], path.stem)
    need(parts and all(part.isidentifier() for part in parts), 'explicit captured Python import name')
    return '.'.join(parts)


class CapturedModuleFinder(importlib.abc.MetaPathFinder):
    """A module in the captured cohort has no checkout or second-root fallback."""

    def __init__(self, root, pins):
        self.root = regular(root, directory=True)
        self.pins = pins
        self.modules = {}
        for relative in sorted(name for name in pins if name.endswith('.py')):
            name = captured_module_name(relative)
            need(name not in self.modules, 'no ambiguous captured Python import name')
            self.modules[name] = relative
        need(self.modules, 'complete captured Python module cohort')
        self.packages = {name for name, relative in self.modules.items()
                         if PurePosixPath(relative).name == '__init__.py'}
        for name in self.modules:
            parents = ['.'.join(name.split('.')[:i]) for i in range(1, len(name.split('.')))]
            need(all(parent in self.packages for parent in parents),
                 'every captured package ancestor has an authenticated initializer')
        self.roots = {name.split('.')[0] for name in self.modules}
        self.loaded = {}

    def owns(self, name):
        return name.split('.')[0] in self.roots

    def checked_path(self, name):
        relative = self.modules[name]
        path = regular(self.root / relative)
        raw = path.read_bytes()
        need(len(raw) == self.pins[relative]['bytes'] and sha(raw) == self.pins[relative]['sha256'],
             'unchanged complete captured module before import: ' + name)
        return path

    def validate_clean_start(self):
        need(not any(self.owns(name) for name in sys.modules),
             'project modules must start in one fresh isolated process')
        for name in self.modules:
            self.checked_path(name)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            source = self.checked_path(fullname)
            self.loaded[fullname] = self.pins[self.modules[fullname]]
            locations = [str(source.parent)] if fullname in self.packages else None
            return importlib.util.spec_from_file_location(fullname, source,
                                                         submodule_search_locations=locations)
        if self.owns(fullname) or fullname.split('.')[0] in ('automation', 'research') \
                or fullname.startswith(('english_', 'bind_b27_', 'test_english_', 'kws_data_contract')):
            raise ModuleNotFoundError('uncaptured project dependency has no fallback: ' + fullname)
        return None

    def verified_imports(self):
        """Reopen every actual ordinary import, including packages and scripts."""
        imported, paths = {}, {}
        for name, pin in sorted(self.loaded.items()):
            module = sys.modules.get(name)
            path = self.checked_path(name)
            need(module is not None and Path(module.__file__).resolve() == path
                 and module.__spec__ is not None and Path(module.__spec__.origin).resolve() == path,
                 'no captured dependency was overwritten by a second-root module: ' + name)
            if name in self.packages:
                need(list(module.__path__) == list(module.__spec__.submodule_search_locations)
                     == [str(path.parent)], 'captured package has one exact original search root: ' + name)
            imported[name] = dict(pin)
            paths[name] = self.modules[name]
        for name in sys.modules:
            if name.split('.')[0] in {value.split('.')[0] for value in self.packages}:
                need(name in imported, 'all original package imports are captured and recorded: ' + name)
        return imported, paths


@contextmanager
def captured_imports(root, pins):
    finder = CapturedModuleFinder(root, pins)
    finder.validate_clean_start()
    before_path, before_meta = list(sys.path), list(sys.meta_path)
    sys.path.insert(0, str(Path(root) / 'automation'))
    sys.meta_path.insert(0, finder)
    try:
        yield finder
    finally:
        sys.path[:] = before_path
        sys.meta_path[:] = before_meta
        # There were no project modules before entry. Removing this isolated
        # cohort prevents an in-process fixture from leaving a false authority.
        for name in tuple(sys.modules):
            if finder.owns(name):
                sys.modules.pop(name, None)


@contextmanager
def bootstrap_offline():
    """No scientific/network/process operation while importing original code."""
    old_import, old_socket = builtins.__import__, socket.socket
    def guarded(name, *args, **kwargs):
        need(name.split('.')[0] not in FORBIDDEN, 'no scientific/network import in original-source bootstrap')
        return old_import(name, *args, **kwargs)
    def blocked(*args, **kwargs):
        raise ValueError('no network or child process during original-source bootstrap')
    class BlockedSocket(old_socket):
        def __init__(self, *args, **kwargs):
            blocked()
    with ExitStack() as stack:
        stack.enter_context(mock.patch('builtins.__import__', side_effect=guarded))
        stack.enter_context(mock.patch('socket.socket', BlockedSocket))
        stack.enter_context(mock.patch('socket.create_connection', side_effect=blocked))
        stack.enter_context(mock.patch('subprocess.Popen', side_effect=blocked))
        yield


def check_original_report(report, lock):
    need(report['schema'] == 'english-v6-b27-verified-original-quality-v1'
         and report['panel'] == 'primary' and report['panel_gate'] == 'NO_PROMOTION'
         and report['original_lock_sha256'] == sha(canonical(lock))
         and report['original_result_sha256'] == ORIGINAL_HASHES['original_result_sha256']
         and report['external_proof_binding_sha256'] == ORIGINAL_HASHES['external_proof_binding_sha256']
         and report['model_pt_sha256']['challenger'] == CANDIDATE_PT
         and report['actual_forwards'] == 996 and report['verified_frame_count'] == 178776,
         'only the actual complete original B27 NO_PROMOTION; never a PASS or replacement outcome')
    need(all(report[key] is True for key in ('all_original_arrays_verified', 'original_source_admission_recomputed',
         'original_producer_git_verified', 'all_original_training_proof_verified', 'complete_original_Q1_failure_verified',
         'old_stops_and_reserves_preserved'))
         and all(report[key] is False for key in ('replication_open_allowed', 'candidate_engineering_validation_allowed',
             'automatic_next_assessment_allowed', 'promotion_allowed', 'product_released'))
         and all(type(report[key]) is int and report[key] == 0
                 for key in ('new_model_calls', 'new_source_calls', 'new_observer_calls')),
         'all unchanged original proofs and failure gates remain mandatory')


def validate_report(result, lock):
    exact_lock(lock)
    need(result['schema'] == REPORT_SCHEMA and result['panel'] == 'primary'
         and result['panel_gate'] == 'NO_PROMOTION'
         and result['original_reader_sha256'] == READER_SHA
         and result['import_scope_adapter_sha256'] == sha(Path(__file__).read_bytes())
         and result['original_report_sha256'] == sha(canonical(result['original_report']))
         and result['original_producer_artifact'] == dict(id=ARTIFACTS['producer'][0],
              bytes=ARTIFACTS['producer'][1], sha256=ARTIFACTS['producer'][2])
         and result['captured_primary_files'] == 479
         and re.fullmatch('[0-9a-f]{64}', result['captured_primary_inventory_sha256']) is not None
         and result['original_failed_continuation_run_id'] == 45643969797,
         'exact separately identified adapter and original B27 proof')
    need(all(result[key] is False for key in ('original_verifier_code_changed',
             'original_path_or_hash_guard_changed', 'proof_context_transferred',
             'checkout_fallback_allowed', 'original_standalone_continuation_succeeded',
             'replication_open_allowed', 'candidate_engineering_validation_allowed',
             'automatic_next_assessment_allowed', 'promotion_allowed', 'product_released'))
         and all(type(result[key]) is int and result[key] == 0 for key in
             ('new_model_calls', 'new_source_calls', 'new_observer_calls', 'new_optimizer_updates')),
         'read-only failure certification cannot open PASS, retry or scientific execution')
    modules = result['imported_original_modules']
    paths = result['imported_original_module_paths']
    need(type(modules) is dict and READER_NAME in modules and 'english_v6_b27_profile' in modules,
         'exact original reader and profile import identity is recorded')
    need(type(paths) is dict and set(paths) == set(modules)
         and {'kws_data_contract', 'kws_data_contract.identity'} <= set(modules),
         'all ordinary first-party imports include their captured original paths and package identities')
    for name, pin in modules.items():
        need(type(name) is str and all(part.isidentifier() for part in name.split('.'))
             and type(paths[name]) is str and captured_module_name(paths[name]) == name and type(pin) is dict
             and set(pin) == {'bytes', 'sha256'} and type(pin['bytes']) is int and 0 < pin['bytes'] < MAX_BYTES
             and re.fullmatch('[0-9a-f]{64}', pin['sha256']) is not None,
             'complete original imported module pin')
    need(modules[READER_NAME]['sha256'] == READER_SHA, 'recorded original reader hash')
    check_original_report(result['original_report'], lock)
    return result


def verify_worker(directory, lock, proof_root):
    exact_lock(lock)
    directory, proof_root = regular(directory, directory=True), regular(proof_root, directory=True)
    need(not FORBIDDEN & set(sys.modules), 'fresh stdlib-only independent worker')
    original_failure_preflight(directory)
    with tempfile.TemporaryDirectory(prefix='b27-authenticated-import-root-') as temporary:
        root = Path(temporary) / 'original'
        pins = extract_original_root(directory / 'originals/producer.zip', root)
        with captured_imports(root, pins) as finder:
            with bootstrap_offline():
                reader = importlib.import_module(READER_NAME)
                need(Path(reader.__file__).resolve() == root / ('automation/' + READER_NAME + '.py')
                     and sha(Path(reader.__file__).read_bytes()) == READER_SHA,
                     'exact original captured verifier, with no path spoofing')
            # This is the untouched original offline guard and original worker.
            # Its sole profile/_PROOF context is established in the captured
            # root. Its internal second extraction is still independently
            # verified by every original code, Git, source, array and gate check.
            with reader.offline_only(proof_root, 'primary'):
                report = reader.verify_worker(directory, lock, proof_root)
            check_original_report(report, lock)
            imported, imported_paths = finder.verified_imports()
            need(READER_NAME in imported and 'english_v6_b27_profile' in imported,
                 'reader and its sole profile belong to the authenticated captured cohort')
    result = dict(schema=REPORT_SCHEMA, panel='primary', panel_gate='NO_PROMOTION',
        original_reader_sha256=READER_SHA, import_scope_adapter_sha256=sha(Path(__file__).read_bytes()),
        original_producer_artifact=dict(id=ARTIFACTS['producer'][0], bytes=ARTIFACTS['producer'][1],
            sha256=ARTIFACTS['producer'][2]), captured_primary_files=len(pins),
        captured_primary_inventory_sha256=sha(canonical(pins)), imported_original_modules=imported,
        imported_original_module_paths=imported_paths,
        original_report_sha256=sha(canonical(report)), original_report=report,
        original_verifier_code_changed=False, original_path_or_hash_guard_changed=False,
        proof_context_transferred=False, checkout_fallback_allowed=False,
        original_standalone_continuation_succeeded=False, original_failed_continuation_run_id=45643969797,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, new_optimizer_updates=0,
        replication_open_allowed=False, candidate_engineering_validation_allowed=False,
        automatic_next_assessment_allowed=False, promotion_allowed=False, product_released=False)
    return validate_report(result, lock)


def verify_original(directory, lock, proof_root):
    """Always use a fresh process; cached checkout modules cannot enter scope."""
    exact_lock(lock)
    need(not FORBIDDEN & set(sys.modules), 'read-only certification before scientific imports')
    with tempfile.TemporaryDirectory(prefix='b27-rooted-quality-lock-') as temporary:
        path = Path(temporary) / 'lock.json'
        path.write_bytes(canonical(lock))
        raw = subprocess.check_output([sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()),
            '--verify-worker', str(regular(directory, directory=True)), str(path),
            str(regular(proof_root, directory=True))])
    result = document(raw)
    return validate_report(result, lock)


def captured_probe(producer_path, repaired):
    """Real saved code only; no original-proof call, model, audio or gradient."""
    with tempfile.TemporaryDirectory(prefix='b27-captured-import-probe-') as temporary:
        first, second = Path(temporary) / 'first', Path(temporary) / 'second'
        pins = extract_original_root(producer_path, first)
        need(extract_original_root(producer_path, second) == pins, 'two identical complete original snapshots')
        old_path = list(sys.path)
        finder = None
        with ExitStack() as stack:
            stack.callback(lambda: sys.path.__setitem__(slice(None), old_path))
            if repaired:
                finder = stack.enter_context(captured_imports(first, pins))
            else:
                names = CapturedModuleFinder(first, pins).modules
                need(not set(names) & set(sys.modules), 'original failure probe starts with no project module cache')
                stack.callback(lambda: [sys.modules.pop(name, None) for name in names])
                sys.path.insert(0, str(first / 'automation'))
            stack.enter_context(bootstrap_offline())
            reader = importlib.import_module(READER_NAME)
            try:
                primary = reader.load(second / ('automation/' + PRIMARY_NAME + '.py'), PRIMARY_NAME)
            except ValueError as error:
                need(not repaired and str(error) == 'exact local original implementation',
                     'only the actual original cross-root failure is expected')
                return dict(schema=CHECK_SCHEMA, repaired=False, reproduced_original_failure=str(error),
                            new_model_calls=0, new_source_calls=0, new_observer_calls=0)
            need(repaired, 'unrepaired original must reproduce its observed path mismatch')
            spec = primary.protocol()
            producer, workflow = primary.producer_contract(spec, PREPARED_PARENT, second)
            common, decision, source, plan = primary.helpers(spec)
            need(primary.profile is reader.profile and source.profile is reader.profile,
                 'one original profile and ContextVar namespace, with no authority transfer')
            need(primary.profile._PROOF.get() is None, 'metadata check does not create original-proof authority')
            expected = primary.producer_files(spec, producer, workflow, 'primary')
            need(set(expected) == set(pins), 'all original producer files and dynamic resources remain covered')
            need(all(sha((second / name).read_bytes()) == digest for name, digest in expected.items()),
                 'all479 second-extraction producer bytes match frozen original registration')
            imported, imported_paths = finder.verified_imports()
            return dict(schema=CHECK_SCHEMA, repaired=True, original_reader_sha256=READER_SHA,
                original_evaluator_sha256=PRIMARY_SHA, captured_files=len(pins),
                captured_python_files=len(finder.modules), imported_modules=len(imported),
                imported_original_modules=imported, imported_original_module_paths=imported_paths,
                original_protocol_passed=True,
                original_producer_contract_passed=True, original_helpers_and_source_policy_passed=True,
                single_original_profile=True, proof_context_created=False,
                new_model_calls=0, new_source_calls=0, new_observer_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-worker', action='store_true')
    parser.add_argument('--probe-original', action='store_true')
    parser.add_argument('--probe-repaired', action='store_true')
    parser.add_argument('directory', type=Path)
    parser.add_argument('lock', type=Path, nargs='?')
    parser.add_argument('proof_root', type=Path, nargs='?')
    args = parser.parse_args()
    if args.probe_original or args.probe_repaired:
        need(args.probe_original != args.probe_repaired and not args.verify_worker
             and args.lock is None and args.proof_root is None, 'one actual captured-code probe only')
        result = captured_probe(args.directory, args.probe_repaired)
    else:
        need(args.lock is not None and args.proof_root is not None, 'complete original proof and actual lock required')
        lock = document(regular(args.lock).read_bytes())
        result = (verify_worker if args.verify_worker else verify_original)(args.directory, lock, args.proof_root)
    sys.stdout.buffer.write(canonical(result))
