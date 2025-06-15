import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import dateparser
from urllib.parse import urlparse


def fetch_text_from_url(url: str) -> str:
    """Parse article text from URL using BeautifulSoup."""
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


def extract_key_phrases(text: str, n: int = 5) -> list:
    """Extract key phrases from text using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='russian')
    try:
        tfidf = vectorizer.fit_transform([text])
    except ValueError:
        return []
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_terms = sorted(scores, key=lambda x: x[1], reverse=True)
    phrases = [term for term, _ in sorted_terms[:n]]
    return phrases


def search_google_news(phrase: str) -> list:
    """Search Google News RSS for the phrase and return entries."""
    query = requests.utils.quote(phrase)
    url = f"https://news.google.com/rss/search?q={query}&hl=ru&gl=KZ&ceid=KZ:ru"
    feed = feedparser.parse(url)
    return feed.entries


def compute_similarity(text1: str, text2: str) -> float:
    """Return similarity between two texts using SequenceMatcher."""
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def process_search(article_text: str, phrases: list) -> pd.DataFrame:
    records = []
    seen_urls = set()
    for phrase in phrases:
        entries = search_google_news(phrase)
        for entry in entries[:10]:
            link = entry.get('link')
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)
            published = entry.get('published', '')
            date = dateparser.parse(published)
            snippet = entry.get('title', '')
            fetched_text = fetch_text_from_url(link)
            if not fetched_text:
                continue
            similarity = compute_similarity(article_text, fetched_text)
            if similarity < 0.5:
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


st.title('Определение первоисточника публикации')

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
