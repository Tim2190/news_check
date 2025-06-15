# news_check

This project provides a simple Streamlit application to search similar news articles and identify the likely original source.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app locally:
   ```bash
   streamlit run app.py
   ```

3. In the opened page, paste a news text or a link to a news article and press **"Найти первоисточник"**. The application will search Google News RSS feeds, parse found articles and display a table of potential sources ordered by similarity and publication date.

Dependencies are limited to open web sources and RSS feeds only.
