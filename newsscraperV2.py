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
# Ρύθμιση NLTK για stopwords
# ----------------------------
try:
    from nltk.corpus import stopwords
    english_stopwords = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    english_stopwords = set(stopwords.words('english'))

# ----------------------------
# Ρύθμιση Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# Ρυθμίσεις Site & Κατηγοριών
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
# Συνάρτηση Καθαρισμού URL
# ----------------------------
def clean_url(base_url: str, link: str) -> str:
    """
    Συνδυάζει το base_url με ένα σχετικό link χρησιμοποιώντας το urljoin.
    """
    if not link:
        return None
    return link if link.startswith("http") else urljoin(base_url, link)

# ----------------------------
# Συνάρτηση Scraping Άρθρων
# ----------------------------
def scrape_articles(url: str, title_selector: str, link_selector: str, base_url: str = "") -> list:
    """
    Αντλεί τίτλους και URL άρθρων από τη σελίδα που δίνεται, χρησιμοποιώντας τους αντίστοιχους CSS selectors.
    Επιστρέφει μια λίστα με tuples (title, link).
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ανάκτηση τίτλων
        titles = [elem.get_text(strip=True) for elem in soup.select(title_selector)][:10]

        # Ανάκτηση links
        links = [clean_url(base_url, elem.get('href')) for elem in soup.select(link_selector)][:10]

        # Εξασφαλίζουμε ότι ο αριθμός των links είναι ίδιος με των τίτλων
        if len(links) < len(titles):
            links.extend([None] * (len(titles) - len(links)))

        articles = [(title, link) for title, link in zip(titles, links) if title]
        logging.info(f"Scraped {len(articles)} articles from {url}")
        return articles

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error on {url}: {e}")
        return []

# ----------------------------
# Συνάρτηση Λήψης Περιεχομένου και SEO Data
# ----------------------------
def fetch_article_content(url: str) -> tuple:
    """
    Αντλεί το περιεχόμενο του άρθρου και τα SEO στοιχεία (meta title, meta description, meta keywords)
    από το δοσμένο URL.
    Επιστρέφει tuple: (content, meta_title, meta_description, meta_keywords)
    """
    if not url:
        return "No URL provided.", None, None, None

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Προσπάθεια εξαγωγής του περιεχομένου από το <article> tag, εάν υπάρχει
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = article_tag.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        content = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        if not content:
            content = "No relevant content found."

        # Εξαγωγή SEO στοιχείων
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
# Συνάρτηση Ανάλυσης Συναισθήματος
# ----------------------------
def analyze_sentiment(text: str) -> float:
    """
    Αναλύει το συναίσθημα του κειμένου χρησιμοποιώντας το TextBlob.
    Επιστρέφει την polarity ως float με στρογγυλοποίηση σε 3 δεκαδικά.
    """
    if not text or text.startswith("Error"):
        return 0.0
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3)

# ----------------------------
# Συνάρτηση Ανάλυσης SEO
# ----------------------------
def analyze_seo(text: str) -> tuple:
    """
    Υπολογίζει την πυκνότητα λέξεων (keyword density) αφαιρώντας τα κοινά stopwords,
    και υπολογίζει τις μετρικές αναγνωσιμότητας (Flesch Reading Ease και Flesch-Kincaid Grade).
    Επιστρέφει tuple: (keyword_density, readability_scores)
    """
    if not text:
        return {}, {"flesch_reading_ease": "N/A", "flesch_kincaid_grade": "N/A"}

    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {}, {"flesch_reading_ease": "N/A", "flesch_kincaid_grade": "N/A"}

    # Καθαρισμός λέξεων (αφαίρεση σημείων στίξης και μετατροπή σε lowercase)
    cleaned_words = [word.strip(".,!?;:()[]\"'").lower() for word in words]
    # Αφαίρεση stopwords
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
# Συνάρτηση Δημιουργίας DataFrame
# ----------------------------
def build_dataframe(data: list) -> pd.DataFrame:
    """
    Δημιουργεί ένα DataFrame της Pandas από τα συλλεγμένα δεδομένα.
    """
    return pd.DataFrame(data)

# ----------------------------
# Συνάρτηση Εξαγωγής σε CSV
# ----------------------------
def download_csv(df: pd.DataFrame) -> bytes:
    """
    Μετατρέπει το DataFrame σε CSV για εξαγωγή.
    """
    return df.to_csv(index=False).encode('utf-8')

# ----------------------------
# Κύρια Συνάρτηση
# ----------------------------
def main():
    # Ρυθμίσεις Streamlit
    st.set_page_config(page_title="ABC News SEO & Sentiment Dashboard", layout="wide")
    st.title("📊 ABC News SEO & Sentiment Dashboard")

    data = []
    site = sites[0]  # Υποθέτουμε ότι έχουμε μόνο ένα site (ABC News)

    # Ορισμός CSS selectors για κάθε κατηγορία
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

    # Sidebar φίλτρα: επιλογή κατηγορίας και αναζήτηση λέξεων-κλειδιών
    st.sidebar.header("Filters")
    category_filter = st.sidebar.selectbox("Select Category", ["All"] + list(site["categories"].keys()))
    keyword_search = st.sidebar.text_input("Search Keyword in Title", "")

    # Εκτέλεση scraping και συλλογή δεδομένων για κάθε κατηγορία
    for category, url in site["categories"].items():
        # Χρήση του urlparse για ασφαλή εξαγωγή του base URL
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
                "Content": content[:500]  # Προεπισκόπηση περιεχομένου
            })

    # Δημιουργία DataFrame
    df = build_dataframe(data)

    # Εφαρμογή φίλτρων βάσει κατηγορίας και αναζήτησης λέξεων-κλειδιών
    if category_filter != "All":
        df = df[df["Category"] == category_filter]
    if keyword_search:
        df = df[df["Title"].str.contains(keyword_search, case=False, na=False)]

    st.subheader(f"News Articles ({len(df)})")
    st.dataframe(df[["Category", "Title", "Sentiment", "Flesch Reading Ease", "Flesch-Kincaid Grade", "URL"]])

    # Γράφημα κατανομής του Sentiment
    st.subheader("📈 Sentiment Distribution")
    if not df.empty:
        st.bar_chart(df["Sentiment"])
    else:
        st.write("No data available for the selected filters.")

    # Λεπτομέρειες SEO Analysis για κάθε άρθρο
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

    # Δυνατότητα εξαγωγής των δεδομένων σε CSV
    st.subheader("Download Data")
    csv_data = download_csv(df)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name='abc_news_data.csv',
        mime='text/csv'
    )

# ----------------------------
# Εκτέλεση κώδικα όταν τρέχει ως κύριο πρόγραμμα
# ----------------------------
if __name__ == '__main__':
    main()