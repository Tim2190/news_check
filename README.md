# news_check

This project provides a simple Streamlit application to search similar news articles and identify the likely original source.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The requirements file contains only packages compatible with
   Python 3.13. Libraries that depend on PyTorch
   (`sentence-transformers` and `keybert`) are commented out until
   they provide official wheels for Python 3.13.
   When these packages become available you can install them
   manually to get slightly better keyword extraction.

2. Run the app locally:
   ```bash
   streamlit run app.py
   ```

3. In the opened page, paste a news text or a link to a news article and press **"Найти первоисточник"**. The application will search Google News RSS feeds, parse found articles and display a table of potential sources ordered by similarity and publication date.

Dependencies are limited to open web sources and RSS feeds only.

### Optional dependencies

For better keyword extraction you can install `sentence-transformers`
and `keybert` once they support your Python version. Until then the
application will fall back to a simple TF‑IDF based extractor.
The fallback uses unigrams and bigrams with Russian stop words and
is designed not to fail on very short texts.
