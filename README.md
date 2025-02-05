# ABC News SEO & Sentiment Analysis Scraper

## 📌 Overview
This project is a **web scraping and analysis tool** designed to extract, analyze, and visualize news articles from **ABC News**. It performs **sentiment analysis**, **SEO evaluation**, and **readability scoring**, while also providing an interactive **dashboard** using **Streamlit**.

## 🚀 Features
- **Scraping:** Extracts article titles, URLs, and full content from ABC News categories (Politics, Business, Tech).
- **Sentiment Analysis:** Uses **TextBlob** to assess article sentiment.
- **SEO Analysis:** Extracts **meta title, meta description, meta keywords**, and calculates **keyword density**.
- **Readability Metrics:** Computes **Flesch Reading Ease** and **Flesch-Kincaid Grade** for comprehension analysis.
- **Interactive Dashboard:** Uses **Streamlit** to visualize sentiment distribution and SEO insights.
- **Data Export:** Allows **CSV export** of the extracted and analyzed data.

## 🛠️ Installation
Ensure you have **Python 3.7+** installed.

### 1️⃣ Clone the repository
```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install dependencies
```sh
pip install -r requirements.txt
```

## 🔥 How to Run
Run the Streamlit dashboard with:
```sh
streamlit run main.py
```

## 📊 Usage
1. Select the **category** (Politics, Business, Tech) from the sidebar.
2. Optionally **search for keywords** in article titles.
3. View scraped articles with their **sentiment scores**.
4. Analyze **SEO metrics** like keyword density & meta tags.
5. Export results as a **CSV file**.

## 🏗️ Project Structure
```
📂 your-repo-name/
 ├── 📄 main.py             # Main script with scraping, sentiment analysis & SEO features
 ├── 📄 requirements.txt   # Required dependencies
 ├── 📄 README.md          # Documentation (this file)
```

## 🛠️ Technologies Used
- **Python** (Web Scraping & Data Processing)
- **BeautifulSoup** (Extracting HTML content)
- **Requests** (Fetching website data)
- **TextBlob** (Sentiment analysis)
- **NLTK** (Stopwords & Text Cleaning)
- **Textstat** (Readability metrics)
- **Pandas** (Data handling & CSV export)
- **Streamlit** (Dashboard visualization)

## 🔍 Future Enhancements
- ✅ Support for **multiple news sources**.
- ✅ Improved **error handling** for broken links.
- ✅ Implement **historical data storage** & trends analysis.

## 🤝 Contributions
Feel free to **fork** this project and submit pull requests. Contributions are welcome!

## 📝 License
This project is open-source and available under the **MIT License**.

