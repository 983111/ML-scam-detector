import re
import random

HIGH_RISK_TLDS = [
    'tk','ml','ga','cf','gq','pw','click','loan','win',
    'bid','racing','kim','xyz','top','cc','ru','cn'
]

URL_SHORTENERS = [
    'bit.ly','tinyurl','t.co','goo.gl','is.gd','cutt.ly'
]

VERIFIED_DOMAINS = [
    'google.com','apple.com','amazon.com','microsoft.com',
    'paypal.com','github.com','youtube.com'
]

def extract_urls(text):
    return re.findall(r'(https?:\/\/[^\s]+|www\.[^\s]+)', text.lower())

def extract_features(text, manual_score):
    text_lower = text.lower()
    urls = extract_urls(text)

    # ---- TEXT SIGNALS ----
    has_urgency = int(bool(re.search(r'urgent|act now|verify now|suspended', text_lower)))
    has_money = int(bool(re.search(r'free money|lottery|job|income|profit|earn', text_lower)))
    asks_sensitive = int(bool(re.search(r'password|cvv|pin|login|otp', text_lower)))
    off_platform = int(bool(re.search(r'telegram|whatsapp|dm me', text_lower)))

    text_length = len(text)
    exclamations = text.count('!')
    uppercase_ratio = sum(c.isupper() for c in text) / max(len(text), 1)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

    # ---- URL SIGNALS ----
    num_urls = len(urls)
    ip_url = 0
    shortener = 0
    risky_tld = 0
    spoofing = 0
    verified = 0

    for url in urls:
        if re.search(r'\d+\.\d+\.\d+\.\d+', url):
            ip_url = 1
        if any(s in url for s in URL_SHORTENERS):
            shortener = 1
        if any(url.endswith("." + tld) for tld in HIGH_RISK_TLDS):
            risky_tld = 1
        if any(b in url for b in ['paypal','amazon','google']) and \
           not any(v in url for v in VERIFIED_DOMAINS):
            spoofing = 1
        if any(v in url for v in VERIFIED_DOMAINS):
            verified = 1

    url_ratio = num_urls / max(len(text.split()), 1)

    # ---- LEAKAGE-SAFE MANUAL SCORE ----
    manual_score = min(max(manual_score, -20), 20)

    return [
        has_urgency,
        has_money,
        asks_sensitive,
        off_platform,
        text_length,
        exclamations,
        uppercase_ratio,
        digit_ratio,
        num_urls,
        url_ratio,
        ip_url,
        shortener,
        risky_tld,
        spoofing,
        verified,
        manual_score
    ]
