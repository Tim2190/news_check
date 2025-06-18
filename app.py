import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from urllib.parse import urlparse, quote
from datetime import datetime, timedelta
import dateparser
import os
import json
import dotenv

# Загрузка токена из .env
dotenv.load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

HF_EMBEDDING_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


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


def get_embedding(text: str) -> list:
    try:
        response = requests.post(HF_EMBEDDING_URL, headers=HEADERS, json={"inputs": text[:500]})
        print(f"[HF DEBUG] Status: {response.status_code}")
        if response.status_code == 200:
            output = response.json()
            print(f"[HF DEBUG] Output sample: {output[0][:5] if isinstance(output, list) else output}")
            return output[0] if isinstance(output, list) else []
        else:
            print(f"[HF ERROR] {response.text}")
            return []
    except Exception as e:
        print(f"[HF EXCEPTION] {e}")
        return []


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0
    dot = sum(i*j for i, j in zip(a, b))
    norm_a = sum(i*i for i in a) ** 0.5
    norm_b = sum(i*i for i in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)


def search_google_news(text: str) -> list:
    query = quote(text[:100])
    url = f"https://news.google.com/rss/search?q={query}&hl=ru&gl=KZ&ceid=KZ:ru"
    feed = feedparser.parse(url)
    print(f"[RSS DEBUG] Найдено {len(feed.entries)} публикаций по запросу: {query}")
    return feed.entries


def process_semantic_search(article_text: str) -> pd.DataFrame:
    base_embed = get_embedding(article_text)
    if not base_embed:
        st.warning("Embedding пустой. Проверь API или токен.")
        return pd.DataFrame()

    records = []
    seen = set()
    entries = search_google_news(article_text)

    for entry in entries[:20]:
        link = entry.get('link')
        if not link or link in seen:
            continue
        seen.add(link)
        pub_date = dateparser.parse(entry.get('published', ''))
        if not pub_date or (datetime.now() - pub_date.replace(tzinfo=None) > timedelta(days=3)):
            continue
        fetched = fetch_text_from_url(link)
        print(f"[PARSE DEBUG] Длина текста с {link}: {len(fetched)}")
        if not fetched or len(fetched) < 100:
            continue
        cmp_embed = get_embedding(fetched[:500])
        sim = cosine_similarity(base_embed, cmp_embed)
        print(f"[SIM DEBUG] Сходство с '{entry.get('title', '')[:40]}...': {sim:.4f}")
        if sim < 0.3:
            continue
        records.append({
            'Источник': urlparse(link).netloc,
            'Дата': pub_date.strftime('%Y-%m-%d %H:%M') if pub_date else '',
            'Сходство': f"{sim*100:.1f}%",
            'Заголовок': entry.get('title', ''),
            'Ссылка': link,
            'similarity_value': sim,
            'date_value': pub_date
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=['similarity_value', 'date_value'], ascending=[False, True])
        df = df.drop(columns=['similarity_value', 'date_value'])
    return df


# Streamlit UI
st.title('🧠 Семантический поиск похожих публикаций (на ИИ)')
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
        st.info('Получение смыслового представления текста...')
        df = process_semantic_search(raw)
        if df.empty:
            st.warning('Похожие публикации не найдены')
        else:
            st.dataframe(df)
