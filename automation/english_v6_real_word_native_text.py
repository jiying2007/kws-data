"""Stream the original OpenSLR archives; extract only authenticated TRAIN transcripts."""
from hashlib import md5, sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from english_v6_real_word_transcript_scan import SPELLINGS, PAIR, ALONE, normalized

LOCKS={"train-clean-100":"2a93770f6d5c6c964bc36631d331a522",
       "train-clean-360":"c0e676e450a7ff2f54aeade5171606fa",
       "train-other-500":"d1a0fd59409feb2c614ce4d30c387708"}
MIRRORS=("https://www.openslr.org/resources/12/", "https://openslr.trmal.net/resources/12/", "https://openslr.elda.org/resources/12/")


class Digests:
    def __init__(self,stream):
        self.stream=stream
        self.md5=md5()
        self.sha=sha256()
        self.bytes=0
    def read(self,n):
        b=self.stream.read(n)
        self.bytes+=len(b)
        self.md5.update(b)
        self.sha.update(b)
        return b


def scan(part,origin):
    url=origin+part+".tar.gz"
    pipe=subprocess.Popen(["curl","-fLsS","--connect-timeout","20","--speed-time","100","--speed-limit","4096",url],stdout=subprocess.PIPE)
    source=Digests(pipe.stdout)
    flacs=set()
    refs=set()
    leads=[]
    mero=0
    chapters={}
    try:
        with tarfile.open(fileobj=source,mode="r|gz",bufsize=1024*1024) as archive:
            for member in archive:
                if member.name.endswith(".flac") and member.isfile():
                    ident=Path(member.name).stem
                    if not member.name.startswith("LibriSpeech/"+part+"/") or ident in flacs:raise ValueError("duplicate/out-of-partition audio identity")
                    flacs.add(ident)
                if not member.isfile() or not member.name.endswith(".trans.txt"):
                    continue
                if not member.name.startswith("LibriSpeech/"+part+"/") or member.name in chapters:raise ValueError("transcript identity")
                raw=archive.extractfile(member).read()
                chapters[member.name]=sha256(raw).hexdigest()
                for line in raw.decode("utf-8").splitlines():
                    ident,transcription=line.split(maxsplit=1)
                    if not re.fullmatch(r"\d+-\d+-\d+",ident) or ident in refs:raise ValueError("duplicate official transcript ID")
                    refs.add(ident)
                    words=normalized(transcription)
                    mero+=bool(ALONE.search(words))
                    hits=sorted(set(PAIR.findall(words)))
                    if hits:leads.append(dict(id=ident,transcript=transcription,spellings=hits,member=member.name))
        while source.read(1024*1024):
            pass
        if pipe.wait()!=0 or source.md5.hexdigest()!=LOCKS[part] or refs!=flacs or not refs:
            raise ValueError("incomplete official archive or transcript/original identity set")
    finally:
        if pipe.poll() is None:pipe.kill()
        pipe.stdout.close()
        pipe.wait()
    return dict(partition=part,url=url,archive_md5=source.md5.hexdigest(),archive_sha256=source.sha.hexdigest(),
                archive_bytes=source.bytes,official_transcript_rows=len(refs),unique_flac_member_ids=len(flacs),
                transcript_ids_equal_flac_ids=True,chapters=len(chapters),chapter_transcript_sha256=chapters,
                contains_standalone_mero=mero,matched_utterances_by_spelling={w:sum(w in s["spellings"] for s in leads) for w in SPELLINGS},
                transcript_leads=leads,flac_members_decoded=0,model_calls=0)


def run(part,dest):
    if part not in LOCKS:raise ValueError("TRAIN partition")
    for origin in MIRRORS:
        try:
            result=scan(part,origin)
            dest.mkdir(exist_ok=True)
            (dest/(part+".json")).write_text(json.dumps(result,indent=2)+"\n")
            print(part,result["official_transcript_rows"],result["matched_utterances_by_spelling"])
            return
        except (OSError,tarfile.TarError,EOFError,ValueError) as exc:
            print("archive unavailable or incomplete",origin,part,type(exc).__name__,flush=True)
    raise RuntimeError("no complete checksum-matched copy of "+part)


if __name__=="__main__":
    run(sys.argv[1],Path(sys.argv[2]))
