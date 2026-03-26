"""Check Trevor Lewis page contract text and alternative slugs."""
import sys, re
sys.path.insert(0, '.')
import cloudscraper

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
)
BASE = 'https://puckpedia.com/player'

# Check Lewis page in detail
r = scraper.get(f'{BASE}/trevor-lewis', timeout=12)
html = r.text
# Find all relevant patterns
for kw in ['signed to', 'unsigned', 'is an unrestricted', 'no contract', 'contract', 'professional tryout']:
    idx = html.lower().find(kw)
    if idx >= 0:
        snippet = html[max(0,idx-30):idx+200].replace('\n',' ')
        print(f'Lewis [{kw}]: {snippet[:200]}\n')

print('---')
# Chris Tanev slug hunt
for slug in ['christopher-tanev', 'chris-tanev-1', 'christoph-tanev']:
    r2 = scraper.get(f'{BASE}/{slug}', timeout=8)
    print(f'Tanev {slug}: {r2.status_code}')

print('---')
# Joel Eriksson Ek slug hunt
for slug in ['joel-eriksson-ek', 'joel-eriksson', 'joel-ek', 'joelerikssonek', 'joel-eriksson-ek-1']:
    r2 = scraper.get(f'{BASE}/{slug}', timeout=8)
    print(f'Eriksson Ek {slug}: {r2.status_code}')
