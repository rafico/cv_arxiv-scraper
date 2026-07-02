from enum import Enum


class FeedbackAction(str, Enum):
    SAVE = "save"
    SKIP = "skip"
    IGNORE = "ignore"
    SKIMMED = "skimmed"
    PRIORITY = "priority"
    SHARED = "shared"


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
