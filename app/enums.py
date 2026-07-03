from enum import Enum


class FeedbackAction(str, Enum):
    SAVE = "save"
    SKIP = "skip"
    IGNORE = "ignore"
    SKIMMED = "skimmed"
    PRIORITY = "priority"
    SHARED = "shared"


class MatchType(str, Enum):
    """Candidate-generation match categories stored in Paper.match_type.

    "Interest" tags dense-retrieval candidates admitted by the learned
    interest model rather than an author/affiliation/title whitelist hit.
    """

    AUTHOR = "Author"
    AFFILIATION = "Affiliation"
    TITLE = "Title"
    INTEREST = "Interest"


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
