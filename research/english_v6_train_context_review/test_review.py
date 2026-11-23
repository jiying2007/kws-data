"""Exercise offline audio, explicit annotation choices, persistence and JSON export."""
import json
from pathlib import Path
import sys
import os
import shutil
from playwright.sync_api import sync_playwright

folder=Path(sys.argv[1]).resolve()
index=json.loads((folder/'REVIEW_INDEX.json').read_bytes())
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=os.environ.get('CHROME_BIN') or shutil.which('google-chrome') or shutil.which('chromium'),args=['--no-sandbox','--allow-file-access-from-files'])
    page=browser.new_page(viewport={'width':820,'height':700},accept_downloads=True)
    errors=[]
    page.on('pageerror',lambda exc:errors.append(str(exc)))
    page.goto((folder/'review.html').as_uri())
    assert page.title()=='Hey Mero 录音核听' and page.locator('#export').is_disabled()
    assert len(index['originals'])==page.locator('#originals .orig').count()
    if index['windows']:
        assert page.get_by_label('没有唤醒词',exact=True).is_checked() is False
        audio=page.locator('#entry audio').first
        duration=audio.evaluate("""el => new Promise((resolve,reject)=>{el.onloadedmetadata=()=>resolve(el.duration);el.onerror=()=>reject('bad audio');el.load()})""")
        assert duration>0
        page.get_by_label('听不清 / 不确定',exact=True).check()
        page.reload()
        assert page.get_by_label('听不清 / 不确定',exact=True).is_checked()
        with page.expect_download() as d: page.locator('#export').click()
        result=json.loads(Path(d.value.path()).read_bytes())
        assert result['decisions'][0]['decision']=='U' and all(x['decision'] is None for x in result['decisions'][1:])
        assert result['decisions'][0]['start_sample']==index['windows'][0]['start_sample']
        if len(index['windows'])>1:
            page.locator('#next').click()
            assert page.locator('#picker').evaluate('x=>x.selectedIndex')==1
        page.evaluate('localStorage.clear()')
    page.set_viewport_size({'width':375,'height':720})
    page.reload()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    browser.close()
print('Offline review controls, source audio, JSON identity and mobile layout passed.')
