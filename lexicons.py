"""
Lexicons used by the D (demographic imbalance) dimension.

METHODOLOGICAL NOTE — read before citing D in the paper.
--------------------------------------------------------
This lexicon operationalises gender as a binary. That is a real limitation and
must be stated in the manuscript, not hidden: a paper that criticises
benchmarks for culturally narrow operationalisations cannot silently adopt a
binary-gender measure of its own. Two consequences follow.

  1. D measures *binary-gendered reference imbalance*, not "demographic
     imbalance" in general. Name the dimension accordingly in the paper.
  2. D is symmetric: a female-dominant corpus scores identically to a
     male-dominant one at the same distance from parity. Justify this
     explicitly or replace with a signed measure.

The lexicon is intentionally small, closed-class and pronoun/kinship-based.
Occupation terms are excluded on purpose: including them would entangle D with
the very occupational-stereotype signal that WinoBias measures, reintroducing
circularity through the back door.

Every term is dumped to results/lexicon_audit.csv by script 02 so that
reviewers can inspect exactly what was counted.
"""

FEMALE_TERMS = [
    "she", "her", "hers", "herself",
    "woman", "women", "girl", "girls", "lady", "ladies",
    "mother", "mothers", "mom", "moms", "mum", "mums",
    "daughter", "daughters", "sister", "sisters",
    "wife", "wives", "aunt", "aunts", "niece", "nieces",
    "grandmother", "grandmothers", "grandma", "granddaughter", "granddaughters",
    "female", "females", "mrs", "ms", "madam",
]

MALE_TERMS = [
    "he", "him", "his", "himself",
    "man", "men", "boy", "boys", "gentleman", "gentlemen",
    "father", "fathers", "dad", "dads",
    "son", "sons", "brother", "brothers",
    "husband", "husbands", "uncle", "uncles", "nephew", "nephews",
    "grandfather", "grandfathers", "grandpa", "grandson", "grandsons",
    "male", "males", "mr", "sir",
]

# Terms appearing in both lists would silently bias the ratio. Fail loudly.
_overlap = set(FEMALE_TERMS) & set(MALE_TERMS)
if _overlap:
    raise ValueError(f"Gender lexicon overlap detected: {sorted(_overlap)}")

# ----------------------------------------------------------------------------
# Geographic reference sets
# ----------------------------------------------------------------------------
# "Western" set, defined a priori, used only for descriptive reporting.
# It is NOT an input to G, which is a pure entropy measure and therefore
# free of any Western/non-Western value judgement.
WESTERN_ISO2 = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",             # EU-27
    "GB", "US", "CA", "AU", "NZ", "CH", "NO", "IS",
}
