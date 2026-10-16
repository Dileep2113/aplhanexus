"""News & Catalyst Intelligence engine.

Fetches recent announcements, company headlines, and tags them by catalyst type.
Calculates sentiment score and flags high event risk for upcoming earnings or regulatory action.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class NewsItem:
    title: str
    publisher: str
    published_at: str
    link: str
    category: str = "OTHER"   # RESULTS | ORDER | REGULATORY | GUIDANCE | M&A | EXPANSION | OTHER
    sentiment: str = "NEUTRAL" # POSITIVE | NEGATIVE | NEUTRAL


@dataclass
class NewsReport:
    symbol: str
    items: list[NewsItem] = field(default_factory=list)
    sentiment_score: float = 0.0   # -1.0 to +1.0
    sentiment_label: str = "NEUTRAL"
    primary_catalyst: Optional[str] = None
    high_event_risk: bool = False
    event_risk_notes: list[str] = field(default_factory=list)

    def render_summary(self) -> str:
        lines = [
            f"  News for {self.symbol} | Sentiment: {self.sentiment_label} ({self.sentiment_score:+.2f})",
        ]
        if self.primary_catalyst:
            lines.append(f"    Catalyst: {self.primary_catalyst}")
        if self.high_event_risk:
            lines.append(f"    ⚠️ HIGH EVENT RISK: {'; '.join(self.event_risk_notes)}")
        for it in self.items[:3]:
            lines.append(f"    • [{it.category}] ({it.sentiment}) {it.title}")
        return "\n".join(lines)


# Keyword dictionaries
_CATEGORY_KEYWORDS = {
    "RESULTS": ["result", "profit", "earnings", "q1", "q2", "q3", "q4", "pat", "ebitda", "revenue", "quarter"],
    "ORDER": ["order", "contract", "deal", "bags", "wins", "awarded", "procurement"],
    "REGULATORY": ["sebi", "rbi", "court", "probe", "investigation", "penalty", "fda", "tax", "notice", "ed ", "cbi"],
    "GUIDANCE": ["guidance", "target", "outlook", "upgrade", "downgrade", "brokerage"],
    "M&A": ["acquisition", "merger", "stake", "buyout", "demerger", "amalgamation"],
    "EXPANSION": ["capacity", "capex", "expansion", "new plant", "launches", "facility", "global"],
}

_POSITIVE_WORDS = [
    "profit up", "soars", "surges", "jumps", "beat", "higher", "record", "growth",
    "expansion", "contract win", "order win", "approved", "dividend", "buyback",
    "upgrade", "boost", "outperform", "rally", "positive", "lift"
]

_NEGATIVE_WORDS = [
    "profit down", "drops", "slumps", "falls", "misses", "lower", "loss", "decline",
    "probe", "penalty", "notice", "fraud", "investigation", "downgrade", "warning",
    "slashed", "underperform", "plunges", "debt", "resigns", "risk"
]


def _classify_article(title: str) -> tuple[str, str]:
    t_lower = title.lower()

    # Category classification
    category = "OTHER"
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(kw in t_lower for kw in kws):
            category = cat
            break

    # Sentiment classification
    pos_count = sum(1 for kw in _POSITIVE_WORDS if kw in t_lower)
    neg_count = sum(1 for kw in _NEGATIVE_WORDS if kw in t_lower)

    if pos_count > neg_count:
        sentiment = "POSITIVE"
    elif neg_count > pos_count:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return category, sentiment


def _fetch_google_news(symbol: str) -> list[dict]:
    """Fetch live Indian equity news via Google News RSS (unrestricted, zero API key)."""
    import requests
    import xml.etree.ElementTree as ET
    clean_sym = symbol.replace(".NS", "").replace(".BO", "")
    url = f"https://news.google.com/rss/search?q={clean_sym}+NSE+stock+share&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=8)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        articles = []
        for it in root.findall(".//item")[:10]:
            title = it.find("title").text if it.find("title") is not None else ""
            pub = it.find("pubDate").text if it.find("pubDate") is not None else ""
            link = it.find("link").text if it.find("link") is not None else ""
            if title:
                # Extract publisher if in "Headline - Publisher" format
                pub_name = "Financial Press"
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0]
                    pub_name = parts[1]
                articles.append({
                    "title": title,
                    "publisher": pub_name,
                    "pubDate": pub,
                    "link": link,
                })
        return articles
    except Exception:
        return []


def fetch_news(yahoo_symbol: str) -> NewsReport:
    """Fetch recent news and classify impact for a symbol (Yahoo + Google News fallback)."""
    import yfinance as yf

    report = NewsReport(symbol=yahoo_symbol)
    raw_news = []

    # 1. Try Yahoo Finance
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(yahoo_symbol)
            raw_news = t.news or []
    except Exception:
        raw_news = []

    # 2. Fallback to Google News RSS if Yahoo has fewer than 2 headlines
    if len(raw_news) < 2:
        g_news = _fetch_google_news(yahoo_symbol)
        raw_news.extend(g_news)

    items: list[NewsItem] = []
    total_sentiment_val = 0.0

    for article in raw_news:
        content = article.get("content", {})
        title = content.get("title") or article.get("title", "")
        if not title:
            continue

        publisher = ""
        provider = content.get("provider") or {}
        if isinstance(provider, dict):
            publisher = provider.get("displayName", "")
        if not publisher:
            publisher = article.get("publisher", "")

        pub_date = content.get("pubDate") or article.get("pubDate") or ""
        link = ""
        canonical = content.get("canonicalUrl") or {}
        if isinstance(canonical, dict):
            link = canonical.get("url", "")
        if not link:
            link = article.get("link", "")

        cat, sent = _classify_article(title)
        items.append(NewsItem(
            title=title,
            publisher=publisher,
            published_at=pub_date,
            link=link,
            category=cat,
            sentiment=sent,
        ))

        if sent == "POSITIVE":
            total_sentiment_val += 1.0
        elif sent == "NEGATIVE":
            total_sentiment_val -= 1.0

        # Check for event risks
        t_low = title.lower()
        if any(w in t_low for w in ["probe", "cbi", "ed notice", "investigation", "penalty", "sebi bar"]):
            report.high_event_risk = True
            report.event_risk_notes.append(f"Regulatory headline: {title[:70]}")
        if any(w in t_low for w in ["q1 result", "q2 result", "q3 result", "q4 result", "earnings call today"]):
            report.high_event_risk = True
            report.event_risk_notes.append(f"Earnings release/call immediate: {title[:70]}")

    report.items = items
    if items:
        norm_score = total_sentiment_val / len(items)
        report.sentiment_score = round(norm_score, 2)
        if report.sentiment_score > 0.2:
            report.sentiment_label = "POSITIVE"
        elif report.sentiment_score < -0.2:
            report.sentiment_label = "NEGATIVE"
        else:
            report.sentiment_label = "NEUTRAL"

    # Identify primary catalyst
    for it in items:
        if it.category in ("RESULTS", "ORDER", "EXPANSION", "M&A") and it.sentiment != "NEUTRAL":
            report.primary_catalyst = f"{it.category}: {it.title[:80]}"
            break
    if not report.primary_catalyst and items:
        report.primary_catalyst = items[0].title[:80]

    return report


if __name__ == "__main__":
    rep = fetch_news("RELIANCE.NS")
    print(rep.render_summary())
