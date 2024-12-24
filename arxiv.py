import feedparser
import re
import requests
import PyPDF2
import io
import logging
import os
from datetime import datetime
from tqdm import tqdm
import email.utils
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Whitelist configuration
whitelist_title = [
    'Few Shot', 'Low Light', 'MOT', 'Multiple Object Tracking', 'Nerf',
    'Open Set', 'Remote Sensing', 'SAR', 'Satellite', 'Zero Shot'
]

whitelist_affiliations = [
    'Adobe', 'Apple', 'Berkeley', 'ByteDance', 'Caltech', 'California Institute of Technology',
    'Carnegie Mellon University', 'CMU', 'DeepMind', 'Freiburg', 'Google', 'HKUST', 'IDEA-Research',
    'INRIA', 'KAIST', 'Michigan', 'Microsoft', 'MIT', 'Meta', 'Munich', 'Nvidia', 'Oxford',
    'Princeton', 'Stanford', 'Technion', 'Tel-Aviv', 'Toronto', 'Toyota', 'Tsinghua', 'Voxel51',
    'Weizmann', 'Yale', 'Zurich'
]

whitelist_authors = [
    'Andrew Ng', 'Avidan', 'Bengio', 'Belongie', 'Bursuc', 'Christian Wolf',
    'Corso', 'Cremers', 'Darrell', 'Dekel', 'Dollár', 'Farhadi', 'Fei-Fei',
    'Feichtenhofer', 'Geiger', 'Girshick', 'Gkioxari', 'Goodfellow', 'Grauman',
    'Guibas', 'Hammarstrand', 'Hinton', 'Ioffe', 'Ishan Misra', 'Johnson',
    'Kaiming', 'Kanade', 'Kirillov', 'LeCun', 'Leal-Taixe', 'Lior Wolf',
    'Litany', 'Malik', 'Mubarak Shah', 'Pollefeys', 'Ramanan', 'Savarese',
    'Scaramuzza', 'Schmidhuber', 'Sebastian Thrun', 'Snavely', 'Susskind',
    'Svensson', 'Torralba', 'Torr', 'Vedaldi', 'Zisserman'
]

def check_whitelist_match(text_list, whitelist):
    """
    Check if any item in text_list matches items in whitelist with enhanced matching.
    """

    def create_flexible_pattern(term):
        escaped_term = re.escape(term)
        patterns = [
            r'\b' + escaped_term + r'\b',
            r'\b' + escaped_term.replace(r'\ ', '[-\s]') + r'\b',
        ]
        return patterns

    all_patterns = []
    for term in whitelist:
        all_patterns.extend(create_flexible_pattern(term))

    combined_pattern = r'(?:' + '|'.join(all_patterns) + r')'
    all_matches = []
    for text in text_list:
        matches = re.findall(combined_pattern, text, flags=re.IGNORECASE)
        all_matches.extend(matches)

    unique_matches = list(dict.fromkeys(all_matches))
    return unique_matches


def process_paper(entry):
    """
    Process an individual arXiv paper to download PDF, extract text, and match whitelist terms.
    """
    pdf_url = entry.link.replace("/abs/", "/pdf/")

    # Extract publication date
    try:
        if hasattr(entry, 'published'):
            parsed_date = email.utils.parsedate_to_datetime(entry.published)
            formatted_date = parsed_date.strftime("%Y-%m-%d")
        else:
            formatted_date = "Date Unknown"
    except Exception as date_error:
        logging.warning(f"Could not parse date: {date_error}")
        formatted_date = "Date Unknown"

    try:
        # Download PDF
        pdf_response = requests.get(pdf_url)
        pdf_file = io.BytesIO(pdf_response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract text from first page
        combined_text = ""
        for page_num in range(min(1, len(pdf_reader.pages))):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            # Split the text into lines and take the first 12 lines
            lines = text.splitlines()
            combined_text += "\n".join(lines[2:12])

        # Check for whitelist matches
        title_match = check_whitelist_match([entry.title], whitelist_title)
        author_match = check_whitelist_match(entry.author.split(','), whitelist_authors)
        affiliation_match = check_whitelist_match([combined_text], whitelist_affiliations)

        # Determine match priority and type (CHANGED PRIORITY ORDER)
        if author_match:
            match_type = 'Author'
            matching_match = author_match
            match_priority = 1  # Highest priority
        elif affiliation_match:
            match_type = 'Affiliation'
            matching_match = affiliation_match
            match_priority = 2  # Second priority
        elif title_match:
            match_type = 'Title'
            matching_match = title_match
            match_priority = 3  # Lowest priority
        else:
            return None

        return {
            'title': entry.title,
            'authors': entry.author,
            'link': entry.link,
            'pdf_link': pdf_url,
            'matches': matching_match,
            'match_type': match_type,
            'match_priority': match_priority,
            'publication_date': formatted_date
        }
    except Exception as e:
        logging.error(f"Error processing paper: {e}")

    return None


def process_arxiv_papers():
    """
    Process arXiv RSS feed and extract papers matching whitelist in parallel.
    """
    # Ensure logs directory exists
    logs_dir = os.path.expanduser("~/logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create log file with current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(logs_dir, f"papers_{current_date}.txt")

    matching_articles = []
    feed = feedparser.parse("https://rss.arxiv.org/rss/cs.CV")

    logging.info(f"Total entries in RSS feed: {len(feed.entries)}")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for entry in feed.entries:
            futures.append(executor.submit(process_paper, entry))

        for future in tqdm(as_completed(futures), desc="Processing feed entries", unit="entry"):
            result = future.result()
            if result:
                matching_articles.append(result)

    # Sort matching articles by match priority and then by publication date
    sorted_articles = sorted(matching_articles, key=lambda x: (x['match_priority'], x['publication_date']),
                             reverse=False)

    # Open log file and prepare to write results
    with open(log_file_path, 'w') as log_file:
        # Log header
        log_file.write("===== Matched ArXiv Papers =====\n")
        log_file.write(f"Total Matched Articles: {len(sorted_articles)}\n")
        log_file.write("=" * 50 + "\n\n")

        # Print and log matching articles
        print(f"\n===== Matched Articles (logged to {log_file_path}) =====")
        for idx, article in enumerate(sorted_articles, 1):
            output = (
                f"\n{idx}. MATCHED PAPER\n"
                f"Match Type: {article['match_type']}\n"
                f"Title: {article['title']}\n"
                f"Authors: {article['authors']}\n"
                f"ArXiv Link: {article['link']}\n"
                f"PDF Link: {article['pdf_link']}\n"
                f"Publication Date: {article['publication_date']}\n"
            )

            if article['matches']:
                output += "Matched Terms:\n"
                for match in article['matches']:
                    output += f"  - {match}\n"

            output += "-" * 50 + "\n"

            # Write to log file
            log_file.write(output)

            # Print to console
            print(output, end='')

    logging.info(f"Results written to {log_file_path}")
    logging.info(f"Total matched articles: {len(sorted_articles)}")

    return sorted_articles


# Run the script
if __name__ == "__main__":
    process_arxiv_papers()