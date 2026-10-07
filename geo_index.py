"""
Shared geographic surface-form resolution for scripts 04 and 04b.

WHY THIS EXISTS
---------------
The first version of script 04 mapped roughly half of all GPE/LOC entities to
a country and discarded the rest as "unmapped". An audit of the discarded half
showed the failures were not random. They were dominated by:

  * determiner-prefixed country names -- "the United States" failed while
    "United States" succeeded;
  * US sub-national units -- California, Texas, Ohio, Michigan and the rest,
    none of which resolved to US;
  * continents and supranational regions -- Europe, Africa, the Middle East,
    which cannot resolve to a country and should never have been counted as
    failures;
  * NER false positives -- Earth, Mars, AI, Windows, and the tokenisation
    artefact "n't".

The first two are pure defects, and both concern US and Western references, so
the old G understated Western concentration. That is the direction that
weakens the paper's own argument, but a number that is wrong in a convenient
direction is still wrong.

WHAT CHANGED
------------
Every surface form is now classified into exactly one of four buckets, and the
mapping rate is reported over the resolvable denominator rather than over
everything spaCy emitted:

    country   resolved to an ISO2 code -- the only bucket entering H and G
    region    continent or supranational region; genuinely not a country
    nongeo    NER false positive
    unmapped  a real place name the index does not know -- the residual that
              still needs auditing

    map_rate = country / (country + unmapped)

Report all four counts in the paper. The residual is the honest measure of how
much geography the instrument still misses.
"""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_DETERMINER = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_POSSESSIVE = re.compile(r"['\u2019]s$")

# Continents and supranational regions. Not countries, so excluded from the
# entropy calculation rather than counted as mapping failures.
REGION_TERMS = {
    "europe", "africa", "asia", "north america", "south america", "america",
    "central america", "latin america", "oceania", "antarctica", "eurasia",
    "middle east", "far east", "near east", "west", "east", "south", "north",
    "western europe", "eastern europe", "southern europe", "northern europe",
    "sub-saharan africa", "north africa", "east africa", "west africa",
    "southern africa", "central africa", "southeast asia", "south asia",
    "east asia", "central asia", "west asia", "asia pacific", "asia-pacific",
    "caribbean", "scandinavia", "balkans", "baltics", "iberia", "levant",
    "european union", "eu", "nato", "commonwealth", "arctic", "the arctic",
    "pacific", "atlantic", "mediterranean", "gulf", "persian gulf",
    "third world", "global south", "global north", "western world", "the west",
    "eastern bloc", "soviet union", "ussr", "yugoslavia", "czechoslovakia",
}

# Frequent spaCy GPE/LOC false positives observed in the audit.
NONGEO_TERMS = {
    "earth", "mars", "moon", "venus", "jupiter", "saturn", "mercury", "pluto",
    "sun", "heaven", "hell", "eden", "atlantis", "utopia",
    "ai", "windows", "linux", "android", "google", "facebook", "twitter",
    "amazon", "apple", "microsoft", "reddit", "youtube", "internet", "web",
    "n't", "n\u2019t", "s", "t", "u", "co", "inc", "ltd", "llc",
    "connecting", "loading", "home", "world", "state", "states", "city",
    "county", "country", "nation", "downtown", "uptown", "midtown",
    "god", "christ", "jesus", "allah",
}


def normalise(surface: str) -> str:
    """Casefold, collapse whitespace, drop a leading determiner and a
    trailing possessive. 'The United States' and \"the United States's\"
    both become 'united states'."""
    s = _WS.sub(" ", surface).strip().casefold()
    s = _DETERMINER.sub("", s)
    s = _POSSESSIVE.sub("", s)
    return s.strip()


def build_index(verbose: bool = True) -> dict[str, str]:
    """Lowercased surface form -> ISO2 country code."""
    import geonamescache

    gc = geonamescache.GeonamesCache()
    idx: dict[str, str] = {}

    n_country = 0
    for iso2, c in gc.get_countries().items():
        for field in ("name", "iso3", "fips"):
            v = c.get(field)
            if v:
                k = normalise(str(v))
                if k and k not in idx:
                    idx[k] = iso2
                    n_country += 1

    # US states and territories. These were the single largest source of
    # mapping failure in the audit: 'California' and 'Texas' are among the most
    # frequent GPE entities in web text and none of them resolved.
    n_state = 0
    try:
        for st in gc.get_us_states().values():
            for field in ("name", "code"):
                v = st.get(field)
                if v:
                    k = normalise(str(v))
                    if k and k not in idx:
                        idx[k] = "US"
                        n_state += 1
    except Exception:
        pass

    # Cities, largest population wins on collision (Paris FR over Paris TX).
    best: dict[str, tuple[int, str]] = {}
    for city in gc.get_cities().values():
        k = normalise(str(city["name"]))
        if not k:
            continue
        pop = int(city.get("population") or 0)
        if k not in best or pop > best[k][0]:
            best[k] = (pop, city["countrycode"])
    n_city = 0
    for k, (_p, cc) in best.items():
        if k not in idx:
            idx[k] = cc
            n_city += 1

    # Surface forms and demonyms geonamescache does not carry. Deliberately
    # spread beyond Europe and North America: an index that resolves Western
    # demonyms but not others would bias G in exactly the way this module was
    # written to prevent.
    extra = {
        "usa": "US", "u.s.": "US", "u.s.a.": "US", "us": "US",
        "united states of america": "US", "american": "US", "americans": "US",
        "washington dc": "US", "dc": "US", "d.c.": "US",
        "uk": "GB", "u.k.": "GB", "britain": "GB", "great britain": "GB",
        "england": "GB", "scotland": "GB", "wales": "GB",
        "northern ireland": "GB", "british": "GB", "english": "GB",
        "scottish": "GB", "welsh": "GB",
        "turkiye": "TR", "t\u00fcrkiye": "TR", "turkish": "TR",
        "german": "DE", "germans": "DE", "french": "FR", "italian": "IT",
        "spanish": "ES", "portuguese": "PT", "dutch": "NL", "holland": "NL",
        "swedish": "SE", "norwegian": "NO", "danish": "DK", "finnish": "FI",
        "polish": "PL", "greek": "GR", "irish": "IE", "swiss": "CH",
        "austrian": "AT", "belgian": "BE", "czech": "CZ", "hungarian": "HU",
        "romanian": "RO", "ukrainian": "UA", "russian": "RU", "russians": "RU",
        "chinese": "CN", "japanese": "JP", "korean": "KR", "south korean": "KR",
        "north korean": "KP", "indian": "IN", "indians": "IN",
        "pakistani": "PK", "bangladeshi": "BD", "indonesian": "ID",
        "vietnamese": "VN", "thai": "TH", "filipino": "PH", "malaysian": "MY",
        "singaporean": "SG", "nepali": "NP", "sri lankan": "LK",
        "afghan": "AF", "iranian": "IR", "iraqi": "IQ", "syrian": "SY",
        "lebanese": "LB", "israeli": "IL", "palestinian": "PS",
        "jordanian": "JO", "saudi": "SA", "emirati": "AE", "uae": "AE",
        "qatari": "QA", "kuwaiti": "KW", "yemeni": "YE", "omani": "OM",
        "egyptian": "EG", "nigerian": "NG", "kenyan": "KE", "ethiopian": "ET",
        "ghanaian": "GH", "ugandan": "UG", "tanzanian": "TZ",
        "south african": "ZA", "moroccan": "MA", "algerian": "DZ",
        "tunisian": "TN", "libyan": "LY", "sudanese": "SD", "somali": "SO",
        "senegalese": "SN", "ivorian": "CI", "cameroonian": "CM",
        "zimbabwean": "ZW", "zambian": "ZM", "congolese": "CD",
        "brazilian": "BR", "mexican": "MX", "argentine": "AR",
        "argentinian": "AR", "chilean": "CL", "colombian": "CO",
        "peruvian": "PE", "venezuelan": "VE", "cuban": "CU", "bolivian": "BO",
        "ecuadorian": "EC", "uruguayan": "UY", "paraguayan": "PY",
        "canadian": "CA", "australian": "AU", "new zealander": "NZ",
        "kiwi": "NZ", "burmese": "MM", "myanmar": "MM", "cambodian": "KH",
        "laotian": "LA", "mongolian": "MN", "kazakh": "KZ", "uzbek": "UZ",
        "azerbaijani": "AZ", "armenian": "AM", "georgian": "GE",
    }
    n_extra = 0
    for k, v in extra.items():
        nk = normalise(k)
        if nk and nk not in idx:
            idx[nk] = v
            n_extra += 1

    if verbose:
        print(f"Geo index: {len(idx):,} surface forms "
              f"(countries {n_country:,}, US states {n_state}, "
              f"cities {n_city:,}, manual {n_extra})")
        for probe in ("netherlands", "the netherlands", "california",
                      "the united states", "nigeria", "türkiye"):
            print(f"   probe {probe!r:24} -> {idx.get(normalise(probe))}")
    return idx


def classify(surface: str, idx: dict[str, str]) -> tuple[str, str | None]:
    """Return (bucket, iso2). Bucket is country / region / nongeo / unmapped."""
    k = normalise(surface)
    if not k or len(k) < 2:
        return "nongeo", None
    if k in NONGEO_TERMS:
        return "nongeo", None
    if k in REGION_TERMS:
        return "region", None
    cc = idx.get(k)
    if cc:
        return "country", cc
    return "unmapped", None
