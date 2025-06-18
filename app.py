import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from urllib.parse import urlparse, quote
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
import torch
import dateparser
import os
from dotenv import load_dotenv

# Загружаем токен из .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Загружаем модель
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", use_auth_token=HF_TOKEN)


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


def search_google_news(text: str) -> list:
    query = quote(text)
    url = f"https://news.google.com/rss/search?q={query}&hl=ru&gl=KZ&ceid=KZ:ru"
    feed = feedparser.parse(url)
    return feed.entries


def compare_articles(original_text: str, entries: list) -> pd.DataFrame:
    records = []
    original_embedding = model.encode(original_text, convert_to_tensor=True)
    for entry in entries:
        link = entry.get('link')
        if not link:
            continue
        published = entry.get('published', '')
        date = dateparser.parse(published)
        if not date or (datetime.now() - date.replace(tzinfo=None) > timedelta(days=3)):
            continue
        candidate_text = fetch_text_from_url(link)
        if not candidate_text:
            continue
        candidate_embedding = model.encode(candidate_text, convert_to_tensor=True)
        similarity = float(util.pytorch_cos_sim(original_embedding, candidate_embedding)[0][0])
        if similarity < 0.5:
            continue
        records.append({
            'Источник': urlparse(link).netloc,
            'Дата': date.strftime('%Y-%m-%d %H:%M'),
            'Сходство': f"{similarity * 100:.1f}%",
            'Заголовок': entry.get('title', ''),
            'Ссылка': link,
            'similarity_value': similarity,
            'date_value': date
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=['similarity_value', 'date_value'], ascending=[False, True])
        df = df.drop(columns=['similarity_value', 'date_value'])
    return df


# UI
st.title('🧠 Семантический поиск похожих публикаций')
input_url = st.text_input('Ссылка на статью')
input_text = st.text_area('Или вставьте текст публикации')

if st.button('Найти похожие материалы'):
    article_text = input_text.strip()
    if not article_text and input_url:
        st.info('Парсинг статьи...')
        article_text = fetch_text_from_url(input_url)
    if not article_text:
        st.error('Не удалось получить текст для анализа')
    else:
        st.info('Поиск похожих публикаций...')
        results = compare_articles(article_text, search_google_news(article_text))
        if results.empty:
            st.warning('Похожие публикации не найдены')
        else:
            st.dataframe(results)
