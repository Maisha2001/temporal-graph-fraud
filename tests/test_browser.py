import os
import socket
import threading
import time

import pytest
import uvicorn

from temporal_fraud.investigation import create_app
from test_investigation import authored_bundle

pytestmark=pytest.mark.skipif(os.environ.get('FRAUD_BROWSER_TESTS')!='1',reason='Set FRAUD_BROWSER_TESTS=1 with installed Playwright Chromium')


@pytest.fixture
def page(tmp_path):
    from playwright.sync_api import sync_playwright
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(create_app(authored_bundle(tmp_path/'fixture')),host='127.0.0.1',port=port,log_level='error'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    for _ in range(100):
        if server.started:break
        time.sleep(.05)
    assert server.started
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.goto(f'http://127.0.0.1:{port}')
        page.locator('.case').first.wait_for()
        yield page
        browser.close()
    server.should_exit=True;thread.join(timeout=10)
    assert not thread.is_alive()


def test_ranking_selection_and_explicit_outcomes(page):
    from playwright.sync_api import expect
    expect(page.locator('#case-title')).to_have_text('Event 3')
    page.select_option('#model','history_xgboost')
    expect(page.locator('.case').first).to_contain_text('Event 1')
    page.locator('.case').first.click()
    expect(page.locator('#case-title')).to_have_text('Event 1')
    expect(page.locator('#outcome')).to_be_hidden()
    page.check('#outcomes')
    expect(page.locator('#outcome')).to_contain_text('legitimate')
    expect(page.locator('#graph circle')).to_have_count(3)


def test_replay_filter_and_empty_state(page):
    from playwright.sync_api import expect
    page.locator('#minute').fill('1');page.locator('#minute').dispatch_event('input')
    expect(page.locator('.case')).to_have_count(1)
    page.fill('#search','missing')
    expect(page.locator('#cases')).to_contain_text('No saved cases')
    expect(page.locator('#case-detail')).to_be_hidden()
    page.fill('#search','')
    page.locator('#minute').fill('30');page.locator('#minute').dispatch_event('input')
    page.check('#cold')
    expect(page.locator('.case')).to_have_count(1)
    expect(page.locator('#case-title')).to_have_text('Event 2')


def test_export_and_mobile_layout(page):
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    with page.expect_download() as received:
        page.click('#export')
    download=received.value
    assert download.suggested_filename.startswith('case-')
    import json
    value=json.loads(open(download.path(),encoding='utf-8').read())
    assert 'label' not in value['case']
    assert 'authored' in value['notice'].lower()
