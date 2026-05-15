"""
Shared stopword lists for the HCI landscape analysis pipeline.

Three layers of filtering:
1. ACADEMIC_METHODOLOGY — how papers are written (not what they study)
2. NON_ENGLISH — fragments from non-English abstracts
3. NON_HCI_DOMAINS — subject matter from other fields that leaked in
"""

# Academic methodology & writing patterns — these describe HOW papers are
# written, not WHAT is being studied. Removing these crystallizes the
# subject-matter vocabulary.
ACADEMIC_METHODOLOGY = {
    # Study structure
    "study", "studies", "research", "paper", "article", "review",
    "literature", "literature review", "systematic review", "meta analysis",
    "survey", "investigation", "examination", "exploration",
    "aim", "aims", "objective", "objectives", "purpose", "goal", "goals",
    "hypothesis", "hypotheses", "research question", "research questions",

    # Methods / design
    "method", "methods", "methodology", "approach", "approaches",
    "technique", "techniques", "procedure", "procedures",
    "qualitative", "quantitative", "mixed methods", "case study",
    "cross sectional", "sectional", "longitudinal", "randomized",
    "experimental", "quasi experimental", "exploratory", "descriptive",
    "semi structured", "structured", "unstructured",
    "interview", "interviews", "focus group", "focus groups",
    "questionnaire", "questionnaires",
    "observation", "observations", "ethnographic", "ethnography",
    "grounded theory", "action research", "phenomenological",
    "research design", "study design",

    # Data & sampling
    "data", "dataset", "datasets", "sample", "samples", "sampling",
    "participant", "participants", "respondent", "respondents",
    "subject", "subjects", "population", "cohort",
    "collected", "collection", "data collected", "data collection",
    "recruited", "recruited participants", "enrolled",
    "inclusion criteria", "exclusion criteria",
    "sample size", "convenience", "purposive", "snowball",

    # Analysis
    "analysis", "analyses", "analyzed", "analysed",
    "data analyzed", "data analysis", "thematic analysis",
    "content analysis", "statistical analysis", "regression",
    "correlation", "anova", "chi square", "test",
    "coded", "coding", "themes", "thematic", "categories",
    "inductive", "deductive",

    # Results / findings
    "results", "result", "findings", "finding", "showed", "shown",
    "revealed", "indicated", "demonstrated", "suggested", "found",
    "observed", "obtained", "reported", "identified", "confirmed",
    "significant", "significantly", "statistically", "statistically significant",
    "positive", "negative", "higher", "lower", "greater", "less",
    "compared", "comparison", "difference", "differences",
    "associated", "association", "relationship", "relationships",
    "correlated", "correlation", "correlations",
    "increased", "decreased", "improved", "reduced",
    "consistent", "inconsistent",
    "mean", "median", "average", "standard deviation",
    "confidence interval", "value", "values", "score", "scores",
    "prevalence", "incidence", "rate", "rates", "ratio",
    "percentage", "proportion", "frequency",
    "odds ratio", "hazard ratio",

    # Discussion / conclusion
    "discuss", "discussion", "conclude", "conclusion", "conclusions",
    "implication", "implications", "recommendation", "recommendations",
    "limitation", "limitations", "future research", "future studies",
    "further research", "suggest", "suggests",
    "contribute", "contribution", "contributions",
    "highlight", "highlights", "emphasized", "underscore",

    # Generic academic verbs
    "proposed", "propose", "proposes", "present", "presents", "presented",
    "investigate", "investigated", "examine", "examined", "examines",
    "explore", "explored", "explores", "develop", "developed", "develops",
    "improve", "improved", "evaluate", "evaluated", "evaluates",
    "assess", "assessed", "assess", "determine", "determined",
    "apply", "applied", "conduct", "conducted",
    "describe", "described", "report", "reported",
    "compare", "identify", "perform", "performed",
    "demonstrate", "demonstrates", "provide", "provides", "provided",
    "indicate", "indicates", "reveal", "reveals",

    # Generic academic nouns
    "approach", "framework", "model", "models", "concept", "concepts",
    "theory", "theories", "theoretical", "perspective", "perspectives",
    "paradigm", "phenomenon", "phenomena",
    "practice", "practices", "practical",
    "strategy", "strategies", "mechanism", "mechanisms",
    "intervention", "interventions",
    "outcome", "outcomes", "measure", "measures", "measurement",
    "variable", "variables", "parameter", "parameters",
    "criterion", "criteria", "indicator", "indicators",
    "dimension", "dimensions", "aspect", "aspects",

    # Generic connective / filler
    "based", "using", "used", "new", "different", "provide",
    "also", "however", "two", "three", "first", "one", "well",
    "use", "may", "can", "including", "et", "al", "work", "order",
    "show", "find", "like", "make", "way", "need", "set", "high",
    "low", "large", "number", "important", "group", "level",
    "non", "applied", "field", "point", "view",
    "basis", "complex", "components", "traditional",
    "context", "multiple", "term", "terms", "long", "short",
    "ability", "area", "areas", "range", "wide", "variety",
    "type", "types", "characteristics", "according",
    "target", "combined", "control", "active",
    "play", "crucial", "essential", "necessary", "required",
    "requires", "help", "understand", "support", "features",
    "various", "several", "general", "specific", "particular",
    "potential", "effective", "better", "best", "good", "possible",
    "widely", "highly", "especially", "particularly", "overall",
    "mainly", "primarily", "furthermore", "moreover", "addition",
    "finally", "second", "third", "following", "called", "known",
    "total", "existing", "current", "recent", "previous", "related",
    "key", "main", "primary", "address",
    "attention", "focus", "role", "impact", "effect", "effects",
    "factor", "factors", "increase", "decrease", "change", "changes",
    "problem", "problems", "solution", "challenge", "challenges",
    "issue", "issues", "process", "processes",
    "abstract", "conclusion", "purpose", "objective",

    # Academic phrasing patterns
    "paper presents", "paper proposes", "results show",
    "results showed", "results suggest", "results indicate",
    "study aims", "study aimed", "study examined",
    "proposed method", "state art", "state art",
    "experimental results", "achieves", "achieves state",
    "outperforms", "baseline", "benchmark",
    "proposed approach", "proposed framework", "proposed model",
    "real world", "end end", "future work",

    # Publication
    "journal", "published", "volume", "page", "pages",
    "author", "authors", "cited", "references",
    "et al", "et", "al",
}

# Non-English noise — fragments from Turkish, Spanish, Portuguese, French,
# Malay, German, Dutch, Nordic languages
NON_ENGLISH = {
    # Turkish
    'da', 'de', 'na', 'se', 'en', 'la', 'el', 'le', 'un', 'du', 'ne',
    'te', 'ma', 'sa', 've', 'bu', 'ba', 'ke', 'ya', 'bir', 'ile',
    'ara', 'lar', 'nda', 'tir', 'olarak', 'yap', 'kullan',
    'olan', 'ndan', 'oldu', 'ili', 'man', 'nas',
    'mada', 'edilen', 'ekle', 'elde', 'ger', 'ayr', 'lar',
    'al mada', 'elde edilen', 'ger ekle', 'ayr ca', 'al malar',
    'aras ndaki',
    # Spanish / Portuguese
    'que', 'por', 'los', 'las', 'del', 'les', 'des', 'como', 'este',
    'entre', 'sobre', 'una', 'con', 'para', 'objetivo', 'resultados',
    'desde', 'sido', 'tiene', 'cada', 'siendo', 'pero', 'todo',
    'foram', 'foi', 'ncia', 'uma', 'uso', 'rio', 'dos', 'mais',
    'estudo', 'ser', 'tica',
    # French
    'dans', 'une', 'sur', 'pour', 'par', 'est', 'aux', 'sont', 'cette',
    # Malay / Indonesian
    'dan', 'ini', 'ist', 'dalam', 'untuk', 'dengan', 'yang', 'pada',
    'dari', 'akan', 'atau', 'bahwa', 'hasil', 'penelitian', 'dapat',
    'menunjukkan', 'tersebut', 'menjadi', 'terhadap', 'serta',
    'antara', 'telah', 'secara', 'juga', 'tidak', 'lebih', 'melalui',
    'sebuah', 'ada', 'harus', 'oleh', 'mereka', 'itu', 'bagi',
    'sangat', 'masih', 'memiliki', 'digunakan', 'penelitian ini',
    'hasil penelitian', 'menunjukkan bahwa',
    # German / Dutch / Nordic
    'dem', 'och', 'det', 'som', 'zur', 'bei', 'aus', 'als', 'das',
    'die', 'der', 'und', 'den', 'een', 'het', 'van',
    # HTML/formatting / URLs / short fragments
    'sup', 'sub', 'son', 'amp', 'art', 'land', 'men', 'spss',
    'http', 'www', 'org', 'com', 'https', 'url', 'doi',
    'http www', 'www org', 'https doi', 'doi org', 'www com',
    'xmlns', 'href', 'pdf', 'html',
    'lt', 'gt', 'lt gt', 'gt lt', 'amp', 'nbsp', 'quot',
}

# Non-HCI domain vocabulary — subject matter from other fields
NON_HCI_DOMAINS = {
    # Physics / engineering
    'voltage', 'transistor', 'semiconductor', 'diode',
    'antenna', 'electromagnetic', 'microwave',
    'waveguide', 'optical', 'particle', 'particles',
    'thermal', 'temperatures', 'temperature', 'beam', 'liquid',
    'band', 'metal', 'laser', 'acid', 'magnetic', 'spectral',
    'emission', 'frequency', 'electrode', 'dielectric', 'photon',
    'plasma', 'flux', 'resonance', 'polymer', 'membrane',
    'velocity', 'nonlinear', 'coupling', 'finite',
    'turbulence', 'viscosity', 'oxide', 'thickness',
    # Materials science / chemistry
    'nanoparticles', 'xrd', 'diffraction', 'spectroscopy', 'infrared',
    'morphology', 'synthesized', 'crystal', 'absorption', 'electron',
    'microscopy', 'scanning', 'ray', 'sem', 'tem', 'lattice',
    'alloy', 'tensile', 'corrosion', 'metallurgy',
    'catalyst', 'reagent', 'synthesis',
    'crystalline', 'photocatalytic', 'adsorption',
    # Biology / clinical medicine
    'receptor', 'mrna', 'peptide', 'enzyme', 'protein', 'amino',
    'signaling', 'inhibition', 'binding',
    'rats', 'mice', 'vitro', 'vivo', 'murine', 'cellular',
    'cortex', 'neuron', 'synaptic', 'neuronal',
    'dopamine', 'serotonin', 'hippocampal',
    'mutation', 'genotype', 'phenotype', 'allele', 'chromosome',
    'tissue', 'cell culture', 'antibody', 'antigen',
    'apoptosis', 'proliferation', 'differentiation',
    'gene expression', 'transcription', 'dna', 'rna',
    'serum', 'renal', 'hepatic', 'pulmonary', 'surgical',
    'postoperative', 'cohort', 'carcinoma', 'tumor',
    'nurses', 'nursing', 'adolescents',
    # Agriculture / ecology
    'soil', 'crop', 'irrigation', 'agricultural',
    'species', 'habitat', 'biodiversity', 'ecosystem',
    # Geology
    'geologic', 'seismic', 'tectonic',
    # Humanities (non-HCI)
    'political', 'justice', 'war', 'colonial', 'theological',
    'sermon', 'scripture', 'worship', 'liturgy',
    'literary', 'poetry', 'novel', 'narrative fiction',
    'archaeological', 'excavation',
    'teachers', 'teacher', 'pedagogical', 'educators', 'classroom',
    'curriculum', 'schools', 'religious', 'islamic', 'moral',
    'theology', 'higher education', 'literacy',
    'teaching learning', 'school principal',
}

# Combined set for quick lookup
ALL_STOPWORDS = ACADEMIC_METHODOLOGY | NON_HCI_DOMAINS


def is_valid_term(term):
    """Check if a term is valid for inclusion in analysis."""
    import re
    t = term.lower().strip()
    # Must be at least 3 chars
    if len(t) < 3:
        return False
    # Must be alphabetic (with spaces for bigrams)
    if not re.match(r'^[a-z ]+$', t):
        return False
    # Check against all stopwords
    if t in ALL_STOPWORDS:
        return False
    # Check individual words against non-English noise
    if any(p in NON_ENGLISH for p in t.split()):
        return False
    return True
