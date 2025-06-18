import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from urllib.parse import urlparse, quote
from difflib import SequenceMatcher
import dateparser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from datetime import datetime, timedelta
import re
import os
import json
import dotenv

# Загрузка токена из .env
dotenv.load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Встроенные стоп-слова
RUSSIAN_STOPWORDS = [
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
    "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь"
]

def fetch_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, 'html.parser')
        texts = []
        for tag in ['title']:
            t = soup.find(tag)
            if t and t.text:
                texts.append(t.text)
        for meta in ['description', 'og:description', 'og:title']:
            tag = soup.find('meta', attrs={'name': meta}) or soup.find('meta', attrs={'property': meta})
            if tag and tag.get('content'):
                texts.append(tag['content'])
        texts += [p.get_text(strip=True) for p in soup.find_all('p')]
        return '\n'.join(texts)
    except:
        return ""

def summarize_text_with_hf(text: str) -> str:
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text[:1024]}  # BART ограничен по длине
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        result = response.json()
        return result[0]["summary_text"] if isinstance(result, list) else ""
    except:
        return ""

def extract_phrases_tfidf(text: str, n: int = 7) -> list:
    try:
        sentences = [s.strip() for s in text.split('.') if len(s.strip().split()) > 3]
        if len(sentences) < 2:
            sentences = [text, text]
        vectorizer = CountVectorizer(stop_words=RUSSIAN_STOPWORDS, ngram_range=(1, 2))
        counts = vectorizer.fit_transform(sentences)
        tfidf = TfidfTransformer().fit_transform(counts)
        scores = tfidf.toarray().sum(axis=0)
        terms = vectorizer.get_feature_names_out()
        sorted_items = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_items[:n]]
    except:
        return []

def search_google_news(phrase: str) -> list:
    url = f"https://news.google.com/rss/search?q={quote(phrase)}&hl=ru&gl=KZ&ceid=KZ:ru"
    return feedparser.parse(url).entries

def compute_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1, text2).ratio()

def process_search(article_text: str, phrases: list) -> pd.DataFrame:
    records = []
    seen = set()
    for phrase in phrases:
        for entry in search_google_news(phrase)[:10]:
            link = entry.get('link')
            if not link or link in seen:
                continue
            seen.add(link)
            pub_date = dateparser.parse(entry.get('published', ''))
            if not pub_date or (datetime.now() - pub_date.replace(tzinfo=None) > timedelta(days=3)):
                continue
            fetched = fetch_text_from_url(link)
            if not fetched:
                continue
            sim = compute_similarity(article_text, fetched)
            if sim < 0.3:
                continue
            records.append({
                'Источник': urlparse(link).netloc,
                'Дата': pub_date.strftime('%Y-%m-%d %H:%M') if pub_date else '',
                'Сходство': f"{sim*100:.1f}%",
                'Заголовок': entry.get('title', ''),
                'Ссылка': link
            })
    return pd.DataFrame(records)

# Streamlit UI
st.title('📌 ИИ-анализ публикации и поиск похожих новостей')
url = st.text_input('Ссылка на статью')
text = st.text_area('Или вставьте текст публикации')
if st.button('Анализ и поиск'):
    raw = text.strip()
    if not raw and url:
        st.info('Парсинг текста...')
        raw = fetch_text_from_url(url)
    if not raw:
        st.error('Не удалось получить текст')
    else:
        st.info('Суммаризация...')
        summary = summarize_text_with_hf(raw)
        st.write('📝 Суть текста:', summary)
        st.info('Извлечение ключевых фраз...')
        phrases = extract_phrases_tfidf(summary)
        st.write('📌 Ключевые фразы:', ', '.join(phrases))
        st.info('Поиск похожих публикаций...')
        df = process_search(summary, phrases)
        if df.empty:
            st.warning('Похожие публикации не найдены')
        else:
            st.dataframe(df)
