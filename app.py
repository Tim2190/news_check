import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from urllib.parse import quote, urlparse
from datetime import datetime, timedelta
import dateparser
import os
import dotenv
from newspaper import Article

# Загрузка токена из .env
dotenv.load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Endpoints
HF_EMBEDDING_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_SUMMARIZER_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def fetch_text_from_url(url: str) -> str:
    try:
        article = Article(url, language='ru')
        article.download()
        article.parse()
        return article.title + "\n" + article.text
    except Exception as e:
        print(f"[PARSE ERROR] {e}")
        return ""

def summarize_text(text: str) -> str:
    cleaned = text.strip().replace('\n', ' ')
    words = cleaned.split()
    word_count = len(words)
    print(f"[SUMMARY] Word count: {word_count}")

    if word_count < 30:
        print("[SUMMARY FALLBACK] Текст слишком короткий, возвращаем как есть.")
        return cleaned

    if word_count > 800:
        print(f"[SUMMARY TRUNCATED] Исходный текст обрезан с {word_count} до 800 слов.")
        cleaned = ' '.join(words[:800])

    try:
        response = requests.post(HF_SUMMARIZER_URL, headers=HEADERS, json={"inputs": cleaned})
        print(f"[SUMMARY STATUS] {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result and 'summary_text' in result[0]:
                print("[SUMMARY OK]", result[0]['summary_text'][:100])
                return result[0]['summary_text']
            else:
                print(f"[SUMMARY FORMAT ERROR] {result}")
                return cleaned
        else:
            print(f"[SUMMARY ERROR] {response.text}")
            return cleaned
    except Exception as e:
        print(f"[SUMMARY EXCEPTION] {e}")
        return cleaned

def get_embedding(text: str) -> list:
    try:
        response = requests.post(HF_EMBEDDING_URL, headers=HEADERS, json={"inputs": text[:512]})
        print(f"[EMBEDDING STATUS] {response.status_code}")
        if response.status_code == 200:
            return response.json()[0]
        else:
            print(f"[EMBEDDING ERROR] {response.text}")
            return []
    except Exception as e:
        print(f"[EMBEDDING EXCEPTION] {e}")
        return []

def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0
    dot = sum(i * j for i, j in zip(a, b))
    norm_a = sum(i * i for i in a) ** 0.5
    norm_b = sum(i * i for i in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)

def search_google_news(query: str) -> list:
    q = quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=ru&gl=KZ&ceid=KZ:ru"
    feed = feedparser.parse(url)
    print(f"[RSS DEBUG] Найдено {len(feed.entries)} публикаций по запросу: {query}")
    return feed.entries

def process_semantic_search(article_text: str) -> pd.DataFrame:
    print("[PROCESS] Запуск обработки статьи...")
    summary = summarize_text(article_text)
    if not summary:
        st.warning("Не удалось сделать краткое содержание текста")
        return pd.DataFrame()

    base_embed = get_embedding(summary)
    if not base_embed:
        st.warning("Не удалось получить embedding текста")
        return pd.DataFrame()

    records = []
    seen = set()
    entries = search_google_news(summary)

    for entry in entries[:20]:
        link = entry.get('link')
        if not link or link in seen:
            continue
        seen.add(link)
        pub_date = dateparser.parse(entry.get('published', ''))
        if not pub_date or (datetime.now() - pub_date.replace(tzinfo=None) > timedelta(days=5)):
            continue
        fetched = fetch_text_from_url(link)
        print(f"[PARSE DEBUG] {link} -> {len(fetched)} символов")
        if not fetched or len(fetched) < 100:
            continue
        cmp_embed = get_embedding(fetched)
        sim = cosine_similarity(base_embed, cmp_embed)
        print(f"[SIM DEBUG] Сходство с '{entry.get('title', '')[:40]}...': {sim:.4f}")
        if sim < 0.3:
            continue
        records.append({
            'Источник': urlparse(link).netloc,
            'Дата': pub_date.strftime('%Y-%m-%d %H:%M') if pub_date else '',
            'Сходство': f"{sim * 100:.1f}%",
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
st.title('📡 Поиск первоисточника и распространения новости (ИИ)')
url = st.text_input('Вставьте ссылку на статью')
text = st.text_area('Или вставьте текст вручную')
if st.button('🔍 Найти похожие публикации'):
    raw = text.strip()
    if not raw and url:
        st.info('Парсим содержимое по ссылке...')
        raw = fetch_text_from_url(url)
    if not raw:
        st.error('Не удалось получить текст для анализа.')
    else:
        st.info('Анализируем смысл текста и ищем похожее...')
        df = process_semantic_search(raw)
        if df.empty:
            st.warning('Похожие публикации не найдены.')
        else:
            st.dataframe(df)
