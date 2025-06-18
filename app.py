import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from urllib.parse import urlparse, quote
from difflib import SequenceMatcher
import dateparser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from datetime import datetime, timedelta

# Встроенный список русских стоп-слов
RUSSIAN_STOPWORDS = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь"
]

USE_KEYBERT = False  # Отключено — не используется на Python 3.13

def fetch_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
    except Exception:
        return ""
    if response.status_code != 200:
        return ""
    soup = BeautifulSoup(response.text, 'html.parser')
    texts = []
    title = soup.find('title')
    if title and title.text:
        texts.append(title.text)
    for meta in ['description', 'og:description', 'og:title']:
        tag = soup.find('meta', attrs={'name': meta}) or soup.find('meta', attrs={'property': meta})
        if tag and tag.get('content'):
            texts.append(tag['content'])
    paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
    texts.extend(paragraphs)
    return '\n'.join(texts)


def extract_additional_phrases(text: str) -> list:
    """Extract simple patterns from short news text."""
    extra = []
    # фамилия + должность
    for m in re.findall(r"([А-ЯЁ][а-яё]+)\s+(?:[\w-]+\s+)?(президент|премьер|министр|глава|депутат|сенатор|мэр|губернатор|директор|председатель)", text, flags=re.I):
        extra.append(f"{m[0]} {m[1]}")
    # встреча A и B
    for m in re.findall(r"встреча\s+([А-ЯЁ][а-яё]+)\s+и\s+([А-ЯЁ][а-яё]+)", text, flags=re.I):
        extra.append(f"встреча {m[0]} и {m[1]}")
    # визит в город
    for m in re.findall(r"визит\s+в\s+([А-ЯЁ][а-яё]+)", text, flags=re.I):
        extra.append(f"визит в {m}")
    return extra


def filter_noise_phrases(phrases: list) -> list:
    banned_words = {"президента", "казахстана", "владимира"}
    endings = ("а", "я", "у", "ю", "е", "о", "ом", "ой", "ем", "ам", "ях", "ью", "ов", "ев")
    result = []
    for p in phrases:
        words = p.split()
        if len(words) == 1:
            w = words[0].lower()
            if w in banned_words:
                continue
            if len(w) < 5 or w.endswith(endings):
                continue
        result.append(p)
    return result


def extract_key_phrases(text: str, n: int = 5) -> list:
    try:
        if len(text.strip().split()) < 10:
            return []

        sentences = [s.strip() for s in text.split('.') if len(s.strip().split()) > 3]
        if len(sentences) < 2:
            sentences = [text, text]

        vectorizer = CountVectorizer(stop_words=RUSSIAN_STOPWORDS, ngram_range=(1, 2))
        counts = vectorizer.fit_transform(sentences)

        if counts.shape[1] == 0:
            return []

        tfidf = TfidfTransformer().fit_transform(counts)
        scores = tfidf.toarray().sum(axis=0)
        terms = vectorizer.get_feature_names_out()
        sorted_items = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        phrases = [term for term, _ in sorted_items[:n]]
        # manually expand phrases for short texts
        if len(text.split()) < 100:
            extras = extract_additional_phrases(text)
            if extras:
                phrases.extend(extras)
        phrases = list(dict.fromkeys(filter_noise_phrases(phrases)))
        return phrases
    except Exception as e:
        print(f"[ERROR in extract_key_phrases]: {e}")
        return []


def search_google_news(phrase: str) -> list:
    query = quote(phrase)
    url = f"https://news.google.com/rss/search?q={query}&hl=ru&gl=KZ&ceid=KZ:ru"
    feed = feedparser.parse(url)
    return feed.entries


def compute_similarity(text1: str, text2: str) -> float:
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def process_search(article_text: str, phrases: list) -> pd.DataFrame:
    records = []
    seen_urls = set()
    unique_phrases = list({p for p in phrases if len(p.strip()) > 2})

    for phrase in unique_phrases:
        entries = search_google_news(phrase)
        print(f"[🔍 Поиск]: \"{phrase}\" → найдено {len(entries)} результатов")
        for e in entries[:5]:
            print(f"    → {e.get('title')} — {e.get('link')}")
        for entry in entries[:10]:
            link = entry.get('link')
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)
            published = entry.get('published', '')
            if not published:
                continue
            date = dateparser.parse(published)
            if not date:
                continue
            if datetime.now() - date > timedelta(days=3):
                continue
            snippet = entry.get('title', '')
            fetched_text = fetch_text_from_url(link)
            if not fetched_text:
                continue
            similarity = compute_similarity(article_text, fetched_text)
            print(f"→ Заголовок: {snippet}")
            print(f"→ Сходство: {similarity:.3f}")
            if similarity < 0.3:
                continue
            records.append({
                'Источник': urlparse(link).netloc,
                'Дата': date.strftime('%Y-%m-%d %H:%M') if date else '',
                'Сходство': f"{similarity*100:.1f}%",
                'Заголовок': snippet,
                'Ссылка': link,
                'similarity_value': similarity,
                'date_value': date or dateparser.parse('1970-01-01')
            })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(by=['similarity_value', 'date_value'], ascending=[False, True])
    return df.drop(columns=['similarity_value', 'date_value'])


# Streamlit UI
st.title('🔍 Определение первоисточника публикации')

input_url = st.text_input('Ссылка на статью')
input_text = st.text_area('Или вставьте текст публикации')

if st.button('Найти первоисточник'):
    article_text = input_text.strip()
    if not article_text and input_url:
        st.info('Парсинг статьи...')
        article_text = fetch_text_from_url(input_url)
    if not article_text:
        st.error('Не удалось получить текст для анализа')
    else:
        st.info('Извлечение ключевых фраз...')
        phrases = extract_key_phrases(article_text, 7)
        if not phrases:
            st.error('Не удалось извлечь ключевые фразы')
        else:
            st.write('Ключевые фразы:', ', '.join(phrases))
            st.info('Поиск похожих публикаций...')
            results = process_search(article_text, phrases)
            if results.empty:
                st.warning('Похожие публикации не найдены')
            else:
                st.dataframe(results)
