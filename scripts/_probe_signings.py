"""
Probe PuckPedia's actual signing data delivery mechanism.
The /signings list page appears to be JS-rendered.
Strategy: fetch individual /signing/{id} pages which ARE server-rendered,
and probe the JS bundle for the API endpoint used by the list.
"""
import cloudscraper, re, json
from bs4 import BeautifulSoup

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

# 1. Check an individual signing page (known to exist from homepage)
print("=== Individual signing page /signing/10289 ===")
resp = scraper.get("https://puckpedia.com/signing/10289", timeout=20)
html = resp.text
print(f"Status: {resp.status_code}  Len: {len(html)}")

soup = BeautifulSoup(html, "html.parser")
print("Title:", soup.title.string if soup.title else "N/A")

# Extract player name, cap hit, team, year from the signing page
# Look for structured data
meta_desc = soup.find("meta", {"name": "description"})
if meta_desc:
    print("Meta description:", meta_desc.get("content", ""))

# Look for the main content block
main = soup.find("main") or soup.find("article") or soup.find(class_="content")
if main:
    text = main.get_text(separator=" ", strip=True)
    print("Main content (first 400):", text[:400])

# Look for any JSON-LD structured data
for script in soup.find_all("script", type="application/ld+json"):
    print("JSON-LD:", script.string[:300] if script.string else "empty")

# Dollar amounts
dollars = re.findall(r"\$[\d,]+(?:\.\d+)?", html)
print("Dollar amounts found:", dollars[:10])

# Contract length patterns
lengths = re.findall(r"\d+[-\s](?:year|yr)", html, re.I)
print("Contract lengths found:", lengths[:5])

print()

# 2. Try a lower-numbered signing to see if it's from summer 2024
# The IDs 10289-10301 are recent (2025-26 season), try some lower ones
print("=== Trying lower signing IDs to find year range ===")
for sid in [9000, 8000, 7500, 7000, 6500]:
    r = scraper.get(f"https://puckpedia.com/signing/{sid}", timeout=15)
    s = BeautifulSoup(r.text, "html.parser")
    title = s.title.string if s.title else "no title"
    # Extract any date from the page
    dates = re.findall(r"\b(20(?:21|22|23|24|25))[^0-9]", r.text)
    dollars = re.findall(r"\$[\d,]+(?:\.\d+)?", r.text)
    meta = s.find("meta", {"name": "description"})
    desc = meta.get("content", "") if meta else ""
    print(f"  /signing/{sid}: {title[:50]}  | dates={dates[:2]}  "
          f"| dollars={dollars[:2]}  | desc={desc[:80]}")

print()

# 3. Check if there's a Drupal Views JSON endpoint for the signing list
print("=== Trying common Drupal/API endpoints for signing list ===")
for path in [
    "/signings?_format=json",
    "/api/signings",
    "/puckpedia/signings",
    "/signings?type=ufa&year=2024",
    "/signings?year=2024",
]:
    try:
        r = scraper.get(f"https://puckpedia.com{path}", timeout=10)
        content_type = r.headers.get("content-type", "")
        print(f"  {path}: status={r.status_code}  ct={content_type[:40]}  "
              f"len={len(r.text)}  "
              f"preview={r.text[:80]!r}")
    except Exception as e:
        print(f"  {path}: ERROR {e}")
