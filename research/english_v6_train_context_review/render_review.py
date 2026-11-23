"""Build an offline annotation entry point next to the authenticated WAV files."""
from hashlib import sha256
import json
from pathlib import Path
import sys


def render(directory):
    raw=(directory/'REVIEW_INDEX.json').read_bytes()
    index=json.loads(raw)
    checksum=sha256(raw).hexdigest()
    data=json.dumps(index,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template=Path(__file__).with_name('review-template.html').read_text()
    (directory/'review.html').write_text(template.replace('__MANIFEST_SHA__',checksum).replace('__MANIFEST__',data),encoding='utf-8')


if __name__=='__main__':
    render(Path(sys.argv[1]))
