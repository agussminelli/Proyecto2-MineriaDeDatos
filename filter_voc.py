import re
from sklearn.base import BaseEstimator, TransformerMixin


DOCUMENTATION_TERMS = [
    "interviewee",
    "interviewer",
    "informant",
    "respondent",
    "participant",
    "interview",
    "respondent explained",
    "explained that",
    "participant stated",
    "informant reported",
    "the interviewee",
    "the respondent",
    "the informant",
    "the participant",
]

DOCUMENTATION_TERMS_V2 = [
    # Initial terms
    "interview",
    "interviewee",
    "interviewer",
    "informant",
    "respondent",
    "respondant",
    "participant",

    # Interview administration
    "client",
    "no comment",
    "went smoothly",
    "service provided",
    "thanked",
    "comment",
    "opinion"

    # Reporting language
    "respondent explained",
    "participant stated",
    "informant reported",
    "according participant",
    "explained that",

    # About Deathcauses -> so we dont need the word itself
    "deceased died",
    "cause death"
]


class DocumentationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, terms=None):
        self.terms = terms

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Filtering Documentation Vocabulary...")
        terms = (
            DOCUMENTATION_TERMS_V2
            if self.terms is None
            else self.terms
        )

        # Longer phrases first
        terms = sorted(terms, key=len, reverse=True)

        pattern = re.compile(
            r"(?<!\w)(?:"
            + "|".join(re.escape(t) for t in terms)
            + r")(?!\w)",
            flags=re.IGNORECASE,
        )

        return [
            re.sub(r"\s+", " ", pattern.sub(" ", str(text))).strip()
            for text in X
        ]