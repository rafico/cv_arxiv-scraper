# ArXiv Paper Processor
This script processes the arXiv Computer Vision RSS feed to identify and log papers that match predefined criteria based on titles, authors, or affiliations. It is designed for researchers and enthusiasts who want to streamline the discovery of relevant papers in the computer vision domain.

## Features
RSS Feed Parsing: Retrieves and processes papers from the arXiv RSS feed for computer vision.
PDF Extraction: Downloads the paper's PDF and extracts text from the first page for further analysis.
Whitelist Matching:
Titles, authors, and affiliations are matched against a customizable whitelist to prioritize relevant content.
Flexible regex patterns allow for enhanced matching capabilities.
Parallel Processing: Speeds up processing using ProcessPoolExecutor to handle multiple papers concurrently.
Prioritization:
Matches are categorized by Author, Affiliation, or Title with configurable priority.
Results are sorted by priority and publication date.
Detailed Logging:
Matched papers are logged to a daily file under ~/logs, with details such as match type, authors, and links.
Matching terms and metadata are clearly displayed for easy reference.
## Usage
Dependencies:

Install the required libraries using pip install -r requirements.txt.
Ensure the following are installed:
feedparser
PyPDF2
tqdm
requests
selenium
webdriver_manager
Run the Script:

bash
Copy code
python arxiv_processor.py
Output:

A detailed log file is saved in the ~/logs directory.
The script outputs the list of matched papers, sorted by relevance and date, in the console and log file.
## Configuration
Whitelist Terms:
Edit the whitelist_title, whitelist_authors, and whitelist_affiliations lists to match specific keywords, names, or institutions.
Logging:
Logs are saved in ~/logs with filenames formatted as papers_<YYYY-MM-DD>.txt.
## Example Output
yaml
Copy code
===== Matched Articles (logged to ~/logs/papers_2024-12-24.txt) =====

1. MATCHED PAPER
Match Type: Author
Title: A Novel Approach to Few-Shot Learning
Authors: Andrew Ng, et al.
ArXiv Link: https://arxiv.org/abs/1234.5678
PDF Link: https://arxiv.org/pdf/1234.5678
Publication Date: 2024-12-20
Matched Terms:
  - Andrew Ng
--------------------------------------------------

2. MATCHED PAPER
Match Type: Affiliation
Title: Satellite Imagery Analysis with Neural Networks
Authors: Jane Doe, John Smith
ArXiv Link: https://arxiv.org/abs/2345.6789
PDF Link: https://arxiv.org/pdf/2345.6789
Publication Date: 2024-12-19
Matched Terms:
  - Stanford
--------------------------------------------------
## Notes
The script is designed for extensibility; additional RSS feeds or domains can be added by modifying the feedparser.parse() call.
Adjust the logging level in logging.basicConfig() for more verbose output (e.g., DEBUG).
