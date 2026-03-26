"""Check exact page content for a few remaining players."""
import sys, re
sys.path.insert(0, '.')
import cloudscraper

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
)
BASE = 'https://puckpedia.com/player'

checks = [
    ('Trevor Lewis',    'trevor-lewis'),
    ('Joel Eriksson Ek','joel-eriksson-ek'),
    ('Marc-Andre Fleury','marc-andre-fleury'),
    ('Chris Tanev',     'chris-tanev'),
    ('Joel Eriksson Ek','joel-eriksson-ek-1'),
    ('Jeff Skinner',    'jeff-skinner'),
]

for name, slug in checks:
    r = scraper.get(f'{BASE}/{slug}', timeout=12)
    print(f'\n{name} ({slug}): status={r.status_code}  len={len(r.text)}')
    if r.status_code == 200:
        html = r.text
        for kw in ['signed to', 'cap hit', 'contract', 'unsigned', 'retired']:
            idx = html.lower().find(kw)
            if idx >= 0:
                snippet = html[max(0, idx-20):idx+150].replace('\n',' ')
                print(f'  [{kw}]: {snippet[:160]}')
                break
