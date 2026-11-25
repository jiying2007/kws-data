"""Invented small transport fixtures; no API, model, source or observer call."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

PATH=Path(__file__).with_name('english_v6_b24q1_r1_git_transport.py')
SPEC=importlib.util.spec_from_file_location('q1_transport_test_target',PATH)
t=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(t)


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def blob(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def zipped(members):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,raw in members:
            archive.writestr(name,raw)
    return stream.getvalue()


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.values={'commit':dict(sha=t.ORIGINAL_HEAD,commit={'tree':{'sha':t.ORIGINAL_TREE}},typed=1,fixture_only=True),
                     'tree':dict(sha=t.ORIGINAL_TREE,truncated=False,tree=[],fixture_only=True)}
        self.historical={r:encoded(v) for r,v in self.values.items()}
        self.current={r:t.canonical(v) for r,v in self.values.items()}
        hp={r:t.pin(v) for r,v in self.historical.items()};cp={r:t.pin(v) for r,v in self.current.items()}
        old=dict(artifacts={str(i):{'id':i} for i in range(1,15)},api=hp)
        self.binding=dict(schema=t.SCHEMA,bound=True,original_lock_sha256=t.ORIGINAL_LOCK_SHA,
            original_run_id=t.ORIGINAL_RUN,original_head=t.ORIGINAL_HEAD,original_tree=t.ORIGINAL_TREE,
            scientific_execution_authorized=False,
            snapshots={r:dict(member='source_'+r+'.snapshot.json',repository_path=t.SNAPSHOT_ROOT+'source_'+r+'.snapshot.json',**hp[r]) for r in hp},
            live_git=cp,manifest={'member':'SNAPSHOTS.json','bytes':1,'sha256':'0'*64},
            transport=dict(repository=t.REPO,run_id=7777,head='a'*40,prepared_parent=t.PREPARED_PARENT,
                branch=t.BRANCH,workflow=t.WORKFLOW,artifact=dict(id=999,name='english-v6-b24q1-historical-r1-git-metadata-7777',bytes=1,sha256='0'*64),
                api={},producer_files={}))
        workflow=b'name: invented fixture only\n'
        producer={t.WORKFLOW:workflow,**{self.binding['snapshots'][r]['repository_path']:v for r,v in self.historical.items()}}
        self.binding['transport']['producer_files']={name:dict(t.pin(raw),git_blob_sha1=blob(raw)) for name,raw in producer.items()}
        self.patches=[mock.patch.object(t,'HISTORICAL',hp),mock.patch.object(t,'CURRENT',cp),
            mock.patch.object(t,'BINDING',self.root/'binding.json'),mock.patch.object(t,'BINDING_SHA','0'*64),
            mock.patch.object(t,'_original_lock',return_value=old)]
        for patch in self.patches:patch.start()
        manifest=encoded(t._expected_manifest(self.binding))
        self.members={self.binding['snapshots'][r]['member']:v for r,v in self.historical.items()}
        self.members['SNAPSHOTS.json']=manifest
        self.raw_zip=zipped(self.members.items())
        self.binding['manifest'].update(t.pin(manifest))
        self.binding['transport']['artifact'].update(t.pin(self.raw_zip))
        run=dict(id=7777,head_sha='a'*40,run_attempt=1,event='push',head_branch=t.BRANCH,path=t.WORKFLOW,
            status='completed',conclusion='success',repository={'full_name':t.REPO,'private':False},head_repository={'full_name':t.REPO,'private':False})
        a=self.binding['transport']['artifact']
        artifacts=dict(total_count=1,artifacts=[dict(id=a['id'],name=a['name'],size_in_bytes=a['bytes'],digest='sha256:'+a['sha256'],
            expired=False,workflow_run={'id':7777,'head_sha':'a'*40})])
        commit=dict(sha='a'*40,parents=[{'sha':t.PREPARED_PARENT}],commit={'tree':{'sha':'b'*40}},
            files=[dict(filename=name,status='added',sha=facts['git_blob_sha1']) for name,facts in self.binding['transport']['producer_files'].items()])
        tree=dict(sha='b'*40,truncated=False,tree=[dict(path=name,type='blob',mode='100644',size=facts['bytes'],sha=facts['git_blob_sha1'])
            for name,facts in self.binding['transport']['producer_files'].items()])
        self.origin={r:t.canonical(v) for r,v in dict(run=run,artifacts=artifacts,commit=commit,tree=tree).items()}
        self.binding['transport']['api']={r:t.pin(raw) for r,raw in self.origin.items()}
        self.write_binding()
        self.parent=self.root/'assembled';self.parent.mkdir()

    def tearDown(self):
        for patch in reversed(self.patches):patch.stop()
        self.tmp.cleanup()

    def write_binding(self):
        raw=encoded(self.binding);t.BINDING.write_bytes(raw);t.BINDING_SHA=t.sha(raw)

    def assemble(self):
        return t.assemble(self.parent,self.current,self.origin,self.raw_zip)

    def test_pure_assemble_reopen_and_retain_preserve_exact_historical_bytes(self):
        output=self.root/'retained';output.mkdir()
        with mock.patch.object(t,'gh',side_effect=AssertionError('no network')), \
             mock.patch.object(socket,'socket',side_effect=AssertionError('no sockets')), \
             mock.patch.object(subprocess,'Popen',side_effect=AssertionError('no process')):
            self.assertEqual(self.assemble(),self.historical)
            self.assertEqual(t.verify_transport(self.parent),self.historical)
            self.assertEqual(t.retain_transport(self.parent,output),self.historical)
        directory=self.parent/t.TRANSPORT_DIR
        self.assertEqual({p.name for p in directory.iterdir()},t.DIRECTORY_FILES)
        self.assertNotEqual(self.historical['commit'],self.current['commit'])
        self.assertFalse((self.parent/'ORIGINAL_B24R1_STOP').exists())
        receipt=json.loads((directory/'RECEIPT.json').read_bytes())
        for key in ('part_of_original_fourteen_artifacts','reserialization_performed',
                    'scientific_execution_authorized','byte_equality_between_historical_and_current_claimed',
                    'transport_workflow_raw_body_reopened'):
            self.assertIs(receipt[key],False)

    def test_unbound_or_scientific_grant_is_rejected(self):
        for key,value in (('bound',False),('scientific_execution_authorized',True)):
            bad=copy.deepcopy(self.binding);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):t.validate_binding(bad)

    def test_changed_binding_bytes_rejected_before_parse_or_download(self):
        t.BINDING.write_bytes(t.BINDING.read_bytes()+b' ')
        with mock.patch.object(t,'gh',side_effect=AssertionError('no fetch')),self.assertRaises(ValueError):
            t.fetch(self.parent,self.current)

    def test_old_artifact_cannot_be_relabelled_as_new_transport(self):
        bad=copy.deepcopy(self.binding);bad['transport']['artifact']['id']=1
        with self.assertRaisesRegex(ValueError,'not one of the original fourteen'):t.validate_binding(bad)

    def test_repinned_origin_mutations_fail_provenance(self):
        changes=(('run',lambda v:v.update(run_attempt=2)),('run',lambda v:v.update(run_attempt=True)),
            ('run',lambda v:v.update(conclusion='failure')),('run',lambda v:v.update(head_sha='c'*40)),
            ('run',lambda v:v['head_repository'].update(private=True)),
            ('artifacts',lambda v:v['artifacts'][0].update(expired=True)),
            ('artifacts',lambda v:v.update(total_count=True)),
            ('commit',lambda v:v['parents'][0].update(sha='c'*40)),
            ('commit',lambda v:v['files'].append(dict(filename='registry.json',status='added',sha='c'*40))),
            ('tree',lambda v:v.update(truncated=True)),('tree',lambda v:v['tree'][0].update(sha='c'*40)))
        for role,change in changes:
            origin=copy.deepcopy(self.origin);value=json.loads(origin[role]);change(value);origin[role]=t.canonical(value)
            binding=copy.deepcopy(self.binding);binding['transport']['api'][role]=t.pin(origin[role])
            with self.subTest(role=role,mutation=str(change)),self.assertRaises(ValueError):t._origin(binding,origin)

    def test_complete_typed_json_identity_rejects_bool_number_and_list_changes(self):
        for left,right in ((1,True),(1,1.0),(False,0),(-0.0,0.0),({'x':[1]},{'x':[True]}),([1,2],[2,1])):
            self.assertFalse(t.same_json(left,right))
        self.assertTrue(t.same_json({'a':1,'b':[False,None]},{'b':[False,None],'a':1}))

    def test_strict_parser_rejects_duplicate_keys_nonfinite_and_bad_utf8(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e999}',b'\xff'):
            with self.subTest(raw=raw),self.assertRaises((ValueError,UnicodeDecodeError)):t.document(raw)

    def test_repinned_current_json_semantic_drift_still_rejects(self):
        for value in (True,1.0,'1',2):
            current=copy.deepcopy(self.current);obj=json.loads(current['commit']);obj['typed']=value;current['commit']=t.canonical(obj)
            binding=copy.deepcopy(self.binding);binding['live_git']['commit']=t.pin(current['commit'])
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'type-aware'):t._compare(binding,self.historical,current)

    def test_current_raw_must_be_exact_even_when_json_value_is_same(self):
        changed=dict(self.current);changed['commit']+=b'\n'
        with mock.patch.object(t,'gh',side_effect=AssertionError('no download before raw pin')),self.assertRaises(ValueError):
            t.fetch(self.parent,changed)

    def test_repinned_archive_extra_duplicate_and_path_members_fail_closure(self):
        for name in ('extra.json','../escape.json','SNAPSHOTS.json'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');raw=zipped([*self.members.items(),(name,b'{}')])
            binding=copy.deepcopy(self.binding);binding['transport']['artifact'].update(t.pin(raw))
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'closure'):t._snapshots(binding,raw)

    def test_repinned_archive_symlink_rejected(self):
        members=[]
        for name,raw in self.members.items():
            info=zipfile.ZipInfo(name);info.create_system=3
            info.external_attr=(stat.S_IFLNK|0o777)<<16
            members.append((info,raw))
        raw=zipped(members);binding=copy.deepcopy(self.binding);binding['transport']['artifact'].update(t.pin(raw))
        with self.assertRaisesRegex(ValueError,'regular snapshot member'):t._snapshots(binding,raw)

    def test_repinned_archive_bad_crc_rejected(self):
        raw=bytearray(self.raw_zip)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:item=archive.infolist()[0]
        start=item.header_offset+30+len(item.filename.encode())+len(item.extra);raw[start]^=1;raw=bytes(raw)
        binding=copy.deepcopy(self.binding);binding['transport']['artifact'].update(t.pin(raw))
        with self.assertRaises((ValueError,zipfile.BadZipFile)):t._snapshots(binding,raw)

    def test_rewritten_historical_snapshot_or_false_manifest_is_rejected(self):
        for name,changed in (('source_commit.snapshot.json',self.current['commit']),
                            ('SNAPSHOTS.json',t.canonical(dict(t._expected_manifest(self.binding),part_of_original_fourteen_artifacts=True)))):
            members=dict(self.members);members[name]=changed;raw=zipped(members.items())
            binding=copy.deepcopy(self.binding);binding['transport']['artifact'].update(t.pin(raw))
            if name=='SNAPSHOTS.json':binding['manifest'].update(t.pin(changed))
            with self.subTest(name=name),self.assertRaises(ValueError):t._snapshots(binding,raw)

    def test_exact_directory_closure_rejects_extra_missing_and_symlink_files(self):
        self.assemble();directory=self.parent/t.TRANSPORT_DIR
        extra=directory/'EXTRA.json';extra.write_bytes(b'{}')
        with self.assertRaises(ValueError):t.verify_transport(self.parent)
        extra.unlink();path=directory/'origin_run.json';raw=path.read_bytes();path.unlink()
        with self.assertRaises(ValueError):t.verify_transport(self.parent)
        outside=self.root/'outside';outside.write_bytes(raw);path.symlink_to(outside)
        with self.assertRaises(ValueError):t.verify_transport(self.parent)

    def test_saved_current_zip_and_receipt_tampering_rejected(self):
        self.assemble();directory=self.parent/t.TRANSPORT_DIR
        for name in ('current_commit.json','snapshots.zip','RECEIPT.json','BINDING.json'):
            path=directory/name;raw=path.read_bytes();path.write_bytes(raw+b' ')
            with self.subTest(name=name),self.assertRaises(ValueError):t.verify_transport(self.parent)
            path.write_bytes(raw)

    def test_second_assembly_and_retention_cannot_overwrite(self):
        self.assemble()
        with self.assertRaises(ValueError):self.assemble()
        output=self.root/'retained';output.mkdir();t.retain_transport(self.parent,output)
        with self.assertRaises(ValueError):t.retain_transport(self.parent,output)

    def test_fetch_uses_only_the_bound_existing_origin_and_artifact_endpoints(self):
        responses={'actions/runs/7777':self.origin['run'],'actions/runs/7777/artifacts?per_page=100':self.origin['artifacts'],
            'commits/'+'a'*40:self.origin['commit'],'git/trees/'+'b'*40+'?recursive=1':self.origin['tree'],
            'actions/artifacts/999/zip':self.raw_zip}
        with mock.patch.object(t,'gh',side_effect=lambda endpoint:responses[endpoint]) as gh:
            self.assertEqual(t.fetch(self.parent,self.current),self.historical)
        self.assertEqual([c.args[0] for c in gh.call_args_list],list(responses))


if __name__=='__main__':
    unittest.main()
