"""Original TRAIN readback mutation checks, using complete stdlib array fixtures."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b27_training_readback as reader


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def zip_bytes(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items(): archive.writestr(name, raw)
    return output.getvalue()


def npy(frames, first=1.0):
    header = str(dict(descr='<f4', fortran_order=False, shape=(frames, 4))).encode()
    header += b' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + b'\n'
    body = struct.pack('<4f', first, 0., -1., -2.) * frames
    return b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + body, body


def fixture():
    paths, ids = [], []
    for group_number in range(34):
        group = 'g' + str(group_number).zfill(2)
        width = 2 if group_number < 6 or 22 <= group_number < 25 else 1
        for original in range(width):
            uid = group + '-' + str(original)
            ids.append(uid)
            for gain in (1.0, 0.75, 0.5):
                if original == 0 and gain == 0.5 and group_number in (0, 1, 22, 23, 24, 25, 26): continue
                paths.append(dict(group=group, id=uid, gain=str(gain)))
    source = dict(positive_retention=paths[:-2], retention_additions=[dict(v, teacher='B9') for v in paths[-2:]])
    slots = [dict(prefix=6400,
                  ordinary=dict(id='ordinary', samples=3200, source_group_id='ordinary-group'),
                  positive=dict(id=ids[step % len(ids)], samples=3200, source_group_id=ids[step % len(ids)].split('-')[0]),
                  natural=dict(id='natural', samples=3200, source_group_id='natural-group')) for step in range(192)]
    schedule = reader.expected_forward_schedule(slots, source)
    arrays, index, optimizer, positive, ordinary = {}, [], [], [], []
    for number, row in enumerate(schedule):
        key = 'forward_' + str(number).zfill(5)
        raw, body = npy(row['frames'])
        arrays[key + '.npy'] = raw
        index.append(dict(row, array=key, sha256=reader.sha(body)))
        if row['kind'] == 'positive_retention':
            positive.append(dict(**{key: row[key] for key in ('step', 'id', 'group', 'teacher', 'newly_protected')},
                                 gain=str(row['gain']), unweighted_loss=0.))
        if row['kind'] in ('synthetic_ordinary', 'natural_ordinary'):
            ordinary.append(dict(**{key: row[key] for key in ('step', 'id', 'gain', 'prefix', 'frames')},
                group=row['id'] + '-group', mean=0., maximum=0., active_fraction=0., active_cells=0, cells=row['frames'] * 3))
    for step, slot in enumerate(slots, 1):
        optimizer.append(dict(step=step, prefix=slot['prefix'], ordinary_id=slot['ordinary']['id'],
            positive_id=slot['positive']['id'], natural_id=slot['natural']['id'], loss=0., synthetic_ctc=0.,
            gain_consistency=0., natural_ctc=0., positive_retention=0., ordinary_ceiling=0.,
            ceiling_coefficient=0.08, preclip_gradient_norm=1., gradient_clipped=False))
    array_zip = zip_bytes(arrays)
    reload = npy(index[0]['frames'])[0]
    data = {'COMMON_DRAW.json': encoded(slots),
        'challenger/TRAIN_FORWARD_LOGITS.npz': array_zip,
        'challenger/TRAIN_FORWARD_INDEX.json': encoded(dict(records=index, count=1844,
            arrays_sha256=reader.sha(array_zip), pending_unreturned_forward=None, logits_are_raw=True)),
        'challenger/OPTIMIZER_TRACE.json': encoded(optimizer),
        'challenger/POSITIVE_RETENTION_TRACE.json': encoded(positive),
        'challenger/ORDINARY_MARGIN_TRACE.json': encoded(ordinary),
        'challenger/CHECKPOINT_RELOAD_LOGITS.npz': zip_bytes({'saved.npy': reload, 'reloaded.npy': reload})}
    result = dict(saved_training_observations={})
    repin(data, result)
    return data, result, source


def repin(data, result):
    result['saved_training_observations'] = {name: dict(bytes=len(data['challenger/' + name]),
        sha256=reader.sha(data['challenger/' + name])) for name in reader.OBSERVATIONS}


def lock():
    names = reader.artifact_names(123)
    training = dict(run_id=123, run_head='b' * 40, prepared_parent='a' * 40,
        run_branch=reader.train.BRANCH.removeprefix('refs/heads/'), runtime_workflow=reader.train.WORKFLOW,
        result_sha256='c' * 64,
        model_pt_sha256=dict(B0=reader.train.b24.old.B0_PT, B9=reader.train.b24.old.b10.b9.MODELS['B9_control']['pt'],
            control=reader.train.CONTROL_PT, challenger='d' * 64),
        artifacts={role: dict(id=number + 1, bytes=100, sha256=str(number + 1) * 64, name=names[role])
                   for number, role in enumerate(reader.ARTIFACT_ROLES)})
    return dict(schema=reader.LOCK_SCHEMA, training=training, training_producer_manifest_sha256='e' * 64)


class SavedReturnContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()

    def verify(self, members, result, source):
        with zipfile.ZipFile(io.BytesIO(zip_bytes(members))) as archive:
            return reader.saved_training_arrays(archive, result, source)

    def test_complete1844_arrays_two_reload_and_all192_updates(self):
        value = self.verify(*self.original)
        self.assertEqual(value['optimization_arrays'], 1844)
        self.assertEqual(value['checkpoint_reload_arrays'], 2)
        self.assertEqual(value['optimizer_trace_entries'], 192)
        self.assertEqual((value['positive_trace_entries'], value['ordinary_trace_entries']), (692, 768))

    def test_unknown_partial_wrong_draw_cropped_frame_and_inconsistent_loss_reject(self):
        for kind in ('unknown', 'partial', 'draw', 'frames', 'loss', 'positive', 'ordinary'):
            data, result, source = copy.deepcopy(self.original)
            index = json.loads(data['challenger/TRAIN_FORWARD_INDEX.json'])
            if kind == 'unknown': index['pending_unreturned_forward'] = dict(step=192)
            if kind == 'partial': index['records'].pop()
            if kind == 'draw': index['records'][0]['id'] = 'replacement'
            if kind == 'frames': index['records'][0]['frames'] -= 1
            if kind == 'loss':
                trace = json.loads(data['challenger/OPTIMIZER_TRACE.json']); trace[0]['loss'] = 1.
                data['challenger/OPTIMIZER_TRACE.json'] = encoded(trace)
            if kind == 'positive':
                trace = json.loads(data['challenger/POSITIVE_RETENTION_TRACE.json']); trace[0]['teacher'] = 'B0'
                data['challenger/POSITIVE_RETENTION_TRACE.json'] = encoded(trace)
            if kind == 'ordinary':
                trace = json.loads(data['challenger/ORDINARY_MARGIN_TRACE.json']); trace[0]['cells'] -= 1
                data['challenger/ORDINARY_MARGIN_TRACE.json'] = encoded(trace)
            data['challenger/TRAIN_FORWARD_INDEX.json'] = encoded(index)
            repin(data, result)
            with self.assertRaises(ValueError, msg=kind): self.verify(data, result, source)

    def test_missing_array_nonfinite_payload_and_changed_reload_reject_after_repin(self):
        for kind in ('missing', 'nan', 'reload'):
            data, result, source = copy.deepcopy(self.original)
            index = json.loads(data['challenger/TRAIN_FORWARD_INDEX.json'])
            if kind == 'reload':
                frames = index['records'][0]['frames']
                data['challenger/CHECKPOINT_RELOAD_LOGITS.npz'] = zip_bytes({
                    'saved.npy': npy(frames)[0], 'reloaded.npy': npy(frames, 2.)[0]})
            else:
                with zipfile.ZipFile(io.BytesIO(data['challenger/TRAIN_FORWARD_LOGITS.npz'])) as archive:
                    arrays = {name: archive.read(name) for name in archive.namelist()}
                if kind == 'missing': del arrays['forward_00000.npy']
                else:
                    raw, body = npy(index['records'][0]['frames'], float('nan'))
                    arrays['forward_00000.npy'] = raw; index['records'][0]['sha256'] = reader.sha(body)
                data['challenger/TRAIN_FORWARD_LOGITS.npz'] = zip_bytes(arrays)
                index['arrays_sha256'] = reader.sha(data['challenger/TRAIN_FORWARD_LOGITS.npz'])
                data['challenger/TRAIN_FORWARD_INDEX.json'] = encoded(index)
            repin(data, result)
            with self.assertRaises(ValueError, msg=kind): self.verify(data, result, source)

    def test_same_122_paths_and_entire43_originals_required(self):
        data, _result, source = copy.deepcopy(self.original)
        source['positive_retention'].pop()
        with self.assertRaises(ValueError): reader.expected_forward_schedule(json.loads(data['COMMON_DRAW.json']), source)
        with self.assertRaises(ValueError): reader.sequence(b'[NaN]')
        with self.assertRaises(ValueError): reader.sequence(b'[{"x":1,"x":2}]')


class OriginalLockContracts(unittest.TestCase):
    def test_exact_control_and_bounded_original_artifacts(self):
        reader.validate_lock(lock())
        for kind in ('control', 'candidate', 'cap', 'name', 'branch', 'same_parent', 'duplicate_artifact'):
            value = lock(); t = value['training']
            if kind == 'control': t['model_pt_sha256']['control'] = reader.train.INITIAL_PT
            if kind == 'candidate': t['model_pt_sha256']['challenger'] = reader.train.INITIAL_PT
            if kind == 'cap': t['artifacts']['producer']['bytes'] = 128_000_001
            if kind == 'name': t['artifacts']['models']['name'] = 'different-run'
            if kind == 'branch': t['run_branch'] = 'main'
            if kind == 'same_parent': t['prepared_parent'] = t['run_head']
            if kind == 'duplicate_artifact': t['artifacts']['producer']['id'] = t['artifacts']['models']['id']
            with self.assertRaises(ValueError, msg=kind): reader.validate_lock(value)

    def test_loaded_tensor_runtime_rejects_before_any_process_or_file(self):
        with patch.dict(sys.modules, {'torch': object()}), patch.object(reader.subprocess, 'check_output') as spawn:
            with self.assertRaisesRegex(ValueError, 'runtime imports'):
                reader.verify_training_original('/fixture', lock(), '/originals')
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
