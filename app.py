import streamlit as st
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from urllib.parse import urlparse, quote
from difflib import SequenceMatcher
import dateparser

# Попробуем загрузить KeyBERT, иначе используем TF-IDF
USE_KEYBERT = False
try:
    from keybert import KeyBERT
    keybert_model = KeyBERT('all-MiniLM-L6-v2')
    USE_KEYBERT = True
except ImportError:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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
    """Extract key phrases using KeyBERT or TF-IDF fallback."""
    try:
        if USE_KEYBERT:
            keywords = keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='russian',
                top_n=n
            )
            return [kw for kw, _ in keywords if kw.strip()]
        else:
            # Fallback: разбиваем текст для TF-IDF
            sentences = [s.strip() for s in text.split('.') if len(s.strip().split()) > 3]
            if len(sentences) < 2:
                sentences = [text, text]  # дублируем для имитации корпуса

            vectorizer = CountVectorizer(stop_words='russian', ngram_range=(1, 2))
            counts = vectorizer.fit_transform(sentences)
            tfidf = TfidfTransformer().fit_transform(counts)
            scores = tfidf.toarray().sum(axis=0)
            terms = vectorizer.get_feature_names_out()
            sorted_items = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
            return [term for term, _ in sorted_items[:n]]
    except Exception as e:
        print(f"[ERROR in extract_key_phrases]: {e}")
        return []


def search_google_news(phrase: str) -> list:
    """Search Google News RSS for the phrase and return entries."""
    query = quote(phrase)
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
