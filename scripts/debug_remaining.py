"""Quick check of remaining 70 estimated players — find easy wins."""
import sys, re, time
sys.path.insert(0, '.')

import cloudscraper
from src.data.contracts_db import get_all_contracts
from src.data.exhaustive_scraper import _puckpedia_slug_variants, _parse_puckpedia

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
)
BASE = 'https://puckpedia.com/player'

all_db = get_all_contracts()
estimated = [(pid, row) for pid, row in all_db.items() if row and row.get('is_estimated')]
print(f"Checking {len(estimated)} estimated players...\n")

found = []
not_found = []

for pid, row in sorted(estimated, key=lambda x: x[1].get('name', '')):
    name = row.get('name', '')
    variants = _puckpedia_slug_variants(name)[:6]  # only first 6 variants

    result = None
    for slug in variants:
        url = f'{BASE}/{slug}'
        try:
            resp = scraper.get(url, timeout=10)
            if resp.status_code == 200:
                c = _parse_puckpedia(resp.text, 2026)
                if c:
                    result = (slug, c['cap_hit'])
                    break
            time.sleep(0.05)
        except Exception:
            pass

    if result:
        found.append((name, result[0], result[1]))
        print(f"  FOUND  {name:<28} slug={result[0]}  ${result[1]:>10,}")
    else:
        not_found.append(name)

print(f"\n--- Summary ---")
print(f"Found:     {len(found)}")
print(f"Not found: {len(not_found)}")
print(f"\nStill missing:")
for n in sorted(not_found):
    print(f"  {n}")
