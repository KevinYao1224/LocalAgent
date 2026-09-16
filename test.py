import httpx

for url in [
    "http://127.0.0.1:11434/api/version",
    "http://[::1]:11434/api/version",
]:
    try:
        r = httpx.get(url, trust_env=False)
        print(url, r.status_code, repr(r.text))
    except Exception as e:
        print(url, type(e).__name__, e)