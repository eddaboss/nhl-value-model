"""Debug why major stars are failing PuckPedia parse."""
import re
import sys
sys.path.insert(0, '.')
import cloudscraper

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
)

players = [
    ('Connor McDavid',  'connor-mcdavid'),
    ('Mitch Marner',    'mitch-marner'),
    ('Jack Eichel',     'jack-eichel'),
    ('Adrian Kempe',    'adrian-kempe'),
    ('Martin Necas',    'martin-necas'),
]

PAT = re.compile(
    r'is signed to (?:a )?(\d+) year[^$]+'
    r'\$[\d,]+ contract with a cap hit of \$([\d,]+) per season',
    re.IGNORECASE,
)

for name, slug in players:
    url = f'https://puckpedia.com/player/{slug}'
    resp = scraper.get(url, timeout=15)
    print(f'\n{name}:  status={resp.status_code}  len={len(resp.text)}')

    if resp.status_code == 200:
        html = resp.text

        # Contract pattern
        m = PAT.search(html)
        if m:
            print(f'  PARSED: cap_hit=${m.group(2)} ({m.group(1)}-year)')
        else:
            print('  PARSE FAILED — searching for clues...')
            for kw in ['signed to', 'cap hit', 'contract', 'salary']:
                idx = html.lower().find(kw)
                if idx >= 0:
                    snippet = html[max(0, idx-30):idx+120].replace('\n', ' ')
                    print(f'    [{kw}]: ...{snippet}...')
                    break

        title_m = re.search(r'<title[^>]*>([^<]+)</title>', html)
        if title_m:
            print(f'  title: {title_m.group(1)[:80]}')
