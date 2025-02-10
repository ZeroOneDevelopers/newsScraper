import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from textblob import TextBlob
import textstat
from collections import Counter
from urllib.parse import urljoin, urlparse
import nltk

# ----------------------------
# Setting up NLTK for stopwords
# ----------------------------
try:
    from nltk.corpus import stopwords
    english_stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    english_stopwords = set(stopwords.words('english'))

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# Site & Category Settings
# ----------------------------
sites = [
    {
        "name": "ABC News",
        "categories": {
            "Politics": "https://abcnews.go.com/Politics",
            "Business": "https://abcnews.go.com/Business",
            "Tech": "https://abcnews.go.com/Technology"
        }
    }
]

# ----------------------------
# URL Cleaning Function
# ----------------------------
def clean_url(base_url: str, link: str) -> str:
    """
    Combines the base_url with a relative link using urljoin.
    """
    if not link:
        return None
    return link if link.startswith("http") else urljoin(base_url, link)

# ----------------------------
# Article Scraping Function
# ----------------------------
def scrape_articles(url: str, title_selector: str, link_selector: str, base_url: str = "") -> list:
    """
    Fetches titles and article URLs from the given page using the corresponding CSS selectors.
    Returns a list of tuples (title, link).
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Fetching titles
        titles = [elem.get_text(strip=True) for elem in soup.select(title_selector)][:10]

        # Fetching  links
        links = [clean_url(base_url, elem.get('href')) for elem in soup.select(link_selector)][:10]

        # Ensure the number of links matches the number of titles
        if len(links) < len(titles):
            links.extend([None] * (len(titles) - len(links)))

        articles = [(title, link) for title, link in zip(titles, links) if title]
        logging.info(f"Scraped {len(articles)} articles from {url}")
        return articles

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error on {url}: {e}")
        return []

# ----------------------------
# Function to Fetch Article Content and SEO Data
# ----------------------------
def fetch_article_content(url: str) -> tuple:
    """
    Fetches the article content and SEO elements (meta title, meta description, meta keywords)
    from the given URL.
    Returns a tuple: (content, meta_title, meta_description, meta_keywords)
    """
    if not url:
        return "No URL provided.", None, None, None

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Attempt to extract content from the <article> tag, if it exists
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = article_tag.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        content = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        if not content:
            content = "No relevant content found."

        # Extracting SEO elements
        meta_title_tag = soup.find("title")
        meta_title = meta_title_tag.get_text(strip=True) if meta_title_tag else "No Title"
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_description = meta_desc_tag["content"] if meta_desc_tag and meta_desc_tag.get("content") else "No Description"
        meta_keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        meta_keywords = meta_keywords_tag["content"] if meta_keywords_tag and meta_keywords_tag.get("content") else "No Keywords"

        return content, meta_title, meta_description, meta_keywords

    except requests.exceptions.Timeout:
        logging.error(f"Timeout error when fetching {url}")
        return "Error: Request timed out.", None, None, None
    except Exception as e:
        logging.error(f"Error fetching content from {url}: {e}")
        return f"Error fetching content: {e}", None, None, None

# ----------------------------
# Sentiment Analysis Function
# ----------------------------
def analyze_sentiment(text: str) -> float:
    """
    Analyzes the sentiment of the text using TextBlob.
    Returns polarity as a float rounded to 3 decimals.
    """
    if not text or text.startswith("Error"):
        return 0.0
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3)

# ----------------------------
# SEO Analysis Function
# ----------------------------
def analyze_seo(text: str) -> tuple:
    """
    Calculates keyword density by removing common stopwords,
    and computes readability metrics (Flesch Reading Ease and Flesch-Kincaid Grade).
    Returns a tuple: (keyword_density, readability_scores)
    """
    if not text:
        return {}, {"flesch_reading_ease": "N/A", "flesch_kincaid_grade": "N/A"}

    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {}, {"flesch_reading_ease": "N/A", "flesch_kincaid_grade": "N/A"}

    # Clean words (remove punctuation and convert to lowercase)
    cleaned_words = [word.strip(".,!?;:()[]\"'").lower() for word in words]
    # Remove stopwords
    filtered_words = [word for word in cleaned_words if word and word not in english_stopwords]
    
    word_freq = Counter(filtered_words)
    keyword_density = {word: round((count / word_count) * 100, 2)
                       for word, count in word_freq.items() if count > 2}

    readability_scores = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
    }
    
    return keyword_density, readability_scores

# ----------------------------
# DataFrame Creation Function
# ----------------------------
def build_dataframe(data: list) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from the collected data.
    """
    return pd.DataFrame(data)

# ----------------------------
# Export to CSV Function
# ----------------------------
def download_csv(df: pd.DataFrame) -> bytes:
    """
    Converts the DataFrame to a CSV file for export.
    """
    return df.to_csv(index=False).encode('utf-8')

# ----------------------------
# Main Function
# ----------------------------
def main():
    # Streamlit settings
    st.set_page_config(page_title="ABC News SEO & Sentiment Dashboard", layout="wide")
    st.title("📊 ABC News SEO & Sentiment Dashboard")

    data = []
    site = sites[0]  # Assuming we have only one site (ABC News)

    # Define CSS selectors for each category
    selectors = {
        "Politics": {
            "title": "h2 a.AnchorLink",
            "link": "h2 a.AnchorLink"
        },
        "Business": {
            "title": "h2.News__Item__Headline, h4.News__title",
            "link": "h2.News__Item__Headline a, h4.News__title a"
        },
        "Tech": {
            "title": "h2 a.AnchorLink",
            "link": "h2 a.AnchorLink"
        }
    }

    # Sidebar filters: select category and search keywords
    st.sidebar.header("Filters")
    category_filter = st.sidebar.selectbox("Select Category", ["All"] + list(site["categories"].keys()))
    keyword_search = st.sidebar.text_input("Search Keyword in Title", "")

    # Scraping and collecting data for each category
    for category, url in site["categories"].items():
        # Use urlparse for safely extracting the base URL
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        articles = scrape_articles(url, selectors[category]["title"], selectors[category]["link"], base_url)
        for title, link in articles:
            content, meta_title, meta_description, meta_keywords = fetch_article_content(link)
            sentiment = analyze_sentiment(content)
            keyword_density, readability_scores = analyze_seo(content)
            data.append({
                "Category": category,
                "Title": title,
                "URL": link,
                "Sentiment": sentiment,
                "Meta Title": meta_title,
                "Meta Description": meta_description,
                "Meta Keywords": meta_keywords,
                "Keyword Density": keyword_density,
                "Flesch Reading Ease": readability_scores["flesch_reading_ease"],
                "Flesch-Kincaid Grade": readability_scores["flesch_kincaid_grade"],
                "Content": content[:500]  # Content preview
            })

    # Create DataFrame
    df = build_dataframe(data)

    # Apply filters based on category and keyword search
    if category_filter != "All":
        df = df[df["Category"] == category_filter]
    if keyword_search:
        df = df[df["Title"].str.contains(keyword_search, case=False, na=False)]

    st.subheader(f"News Articles ({len(df)})")
    st.dataframe(df[["Category", "Title", "Sentiment", "Flesch Reading Ease", "Flesch-Kincaid Grade", "URL"]])

    # Sentiment Distribution Chart
    st.subheader("📈 Sentiment Distribution")
    if not df.empty:
        st.bar_chart(df["Sentiment"])
    else:
        st.write("No data available for the selected filters.")

    # SEO Analysis Details for each article
    st.subheader("🔍 SEO Analysis Details")
    for index, row in df.iterrows():
        st.markdown(f"### {row['Title']}")
        st.write(f"**Category:** {row['Category']}")
        st.write(f"**Sentiment Score:** {row['Sentiment']}")
        st.write(f"**Flesch Reading Ease:** {row['Flesch Reading Ease']}")
        st.write(f"**Flesch-Kincaid Grade:** {row['Flesch-Kincaid Grade']}")
        st.write(f"**Meta Title:** {row['Meta Title']}")
        st.write(f"**Meta Description:** {row['Meta Description']}")
        st.write(f"**Meta Keywords:** {row['Meta Keywords']}")
        st.write(f"**Keyword Density:** {row['Keyword Density']}")
        st.write(f"**Content Preview:** {row['Content']}")
        if row["URL"]:
            st.markdown(f"[Read more]({row['URL']})")
        st.write("---")

    # Option to export data to CSV
    st.subheader("Download Data")
    csv_data = download_csv(df)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name='abc_news_data.csv',
        mime='text/csv'
    )

# ----------------------------
# Execution when running as the main program
# ----------------------------
if __name__ == '__main__':
    main()