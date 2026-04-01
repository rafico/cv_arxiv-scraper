from enum import Enum


class ScrapeStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class FeedbackAction(str, Enum):
    SAVE = "save"
    SKIP = "skip"
    IGNORE = "ignore"
    SKIMMED = "skimmed"
    PRIORITY = "priority"
    SHARED = "shared"


class FeedType(str, Enum):
    ARXIV_RSS = "arxiv_rss"


class ReadingStatus(str, Enum):
    TO_READ = "to_read"
    READING = "reading"
    READ = "read"


class SortOption(str, Enum):
    TRENDING = "trending"
    NEWEST = "newest"
    SAVED = "saved"
    RECOMMENDED = "recommended"
    CITATIONS = "citations"
