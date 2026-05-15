"""
Step 2b: Filter dataset to keep only true HCI papers.

Tightened filter: requires stronger HCI signal and filters out
clinical medicine, biology, and other non-HCI domains that leaked
in through broad search queries.
"""

import json
import os
import re
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_FILE = os.path.join(DATA_DIR, "hci_papers.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "hci_papers_filtered.jsonl")

# Strong HCI signals — terms clearly about computing/interaction/design
HCI_STRONG = {
    "user interface", "user experience", "usability", "hci",
    "human-computer interaction", "human computer interaction",
    "interaction design", "interface design", "ux design",
    "accessibility", "assistive technology",
    "virtual reality", "augmented reality", "mixed reality", "vr", "ar", "xr",
    "brain-computer interface", "bci",
    "haptic", "gesture recognition", "gaze tracking", "eye tracking",
    "touch screen", "touchscreen", "multitouch",
    "voice interface", "voice assistant", "chatbot", "conversational agent",
    "speech recognition", "natural language interface",
    "wearable computing", "wearable device", "smartwatch",
    "ubiquitous computing", "pervasive computing", "iot",
    "tangible interface", "tangible interaction",
    "information visualization", "data visualization",
    "dashboard", "widget", "gui", "graphical user interface",
    "computer-supported cooperative work", "cscw",
    "crowdsourcing", "gamification",
    "prototype", "wireframe", "mockup",
    "think aloud", "usability test", "user study", "user research",
    "interaction technique", "input device",
}

# Moderate HCI signals — relevant but also appear in non-HCI contexts
HCI_MODERATE = {
    "user", "interface", "interaction", "design", "software",
    "computer", "digital", "application", "web", "mobile",
    "display", "visualization", "gesture", "touch", "screen",
    "feedback", "prototype", "evaluation", "task", "participant",
    "survey", "questionnaire", "robot", "sensor", "algorithm",
    "navigation", "menu", "button", "click", "website", "online",
    "privacy", "security", "authentication", "biometric",
    "game", "player", "e-learning",
    "annotation", "recommendation", "search engine",
    "social media", "collaborative", "interactive",
    "smartphone", "tablet", "laptop", "desktop",
    "responsive design", "front end", "frontend", "backend",
    "api", "framework", "toolkit", "sdk",
}

# Terms that indicate clinical medicine (not HCI)
CLINICAL_SIGNALS = {
    "surgery", "surgical", "postoperative", "preoperative",
    "diagnosis", "diagnosed", "prognosis", "mortality",
    "diabetes", "hypertension", "cardiovascular", "cardiac",
    "renal", "hepatic", "liver", "pulmonary", "respiratory",
    "tumor", "cancer", "oncology", "carcinoma", "malignant",
    "cohort", "retrospective", "prospective study",
    "median age", "mean age", "age years",
    "serum", "plasma", "biopsy", "pathology",
    "complications", "comorbidity", "comorbidities",
    "inflammatory", "infection", "infectious",
    "chemotherapy", "radiation therapy", "transplant",
    "intensive care", "icu", "emergency department",
    "mg", "dosage", "dose", "intravenous",
    "odds ratio", "hazard ratio", "confidence interval",
    "body mass index", "bmi",
    "blood pressure", "cholesterol", "glucose",
    "pregnancy", "neonatal", "pediatric",
    "orthopedic", "fracture", "implant",
    "anesthesia", "sedation",
}

# Terms that indicate biology/neuroscience (not HCI)
BIO_SIGNALS = {
    "receptor", "mrna", "peptide", "enzyme", "protein", "amino",
    "membrane", "signaling", "inhibition", "binding",
    "rats", "mice", "vitro", "vivo", "murine",
    "cortex", "neuron", "synaptic", "neuronal",
    "dopamine", "serotonin", "hippocampal", "hippocampus",
    "mutation", "genotype", "phenotype", "allele", "chromosome",
    "tissue", "cell culture", "antibody", "antigen",
    "apoptosis", "proliferation", "differentiation",
    "gene expression", "transcription", "dna", "rna",
    "in vitro", "in vivo", "cell line",
}

# Terms that indicate other non-HCI domains
OTHER_EXCLUDE = {
    # Agriculture / ecology
    "soil", "crop", "irrigation", "agricultural",
    "species", "habitat", "biodiversity", "ecosystem",
    # Geology
    "geologic", "seismic", "tectonic",
    # Materials science / physics / chemistry
    "alloy", "tensile", "corrosion", "metallurgy",
    "catalyst", "reagent", "synthesis", "polymer",
    "voltage", "transistor", "semiconductor", "diode",
    "antenna", "electromagnetic", "microwave",
    "nanoparticles", "oxide", "thickness", "velocity",
    "nonlinear", "coupling", "waveguide", "spectroscopy",
    "diffraction", "lattice", "crystalline", "photon",
    "thermal conductivity", "dielectric", "ferromagnetic",
    "finite element", "turbulence", "viscosity",
    "laser", "optical fiber", "wavelength",
    # Humanities / social sciences (non-HCI)
    "political", "justice", "war", "colonial", "theological",
    "sermon", "scripture", "worship", "liturgy",
    "literary", "poetry", "novel", "narrative fiction",
    "archaeological", "excavation", "artifact",
    # Pure education (not e-learning/HCI)
    "pedagogical", "curriculum", "classroom management",
    "school principal", "teaching profession",
}


def compute_scores(text_lower):
    """Score HCI relevance vs non-HCI domain signals."""
    strong = sum(1 for term in HCI_STRONG if term in text_lower)
    moderate = sum(1 for term in HCI_MODERATE if term in text_lower)
    clinical = sum(1 for term in CLINICAL_SIGNALS if term in text_lower)
    bio = sum(1 for term in BIO_SIGNALS if term in text_lower)
    other = sum(1 for term in OTHER_EXCLUDE if term in text_lower)

    hci_score = strong * 3 + moderate  # strong signals weighted 3x
    exclude_score = clinical + bio + other

    return hci_score, exclude_score, strong


def main():
    kept = 0
    removed = 0
    total = 0

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            total += 1
            paper = json.loads(line.strip())
            abstract = (paper.get("abstract") or "").lower()
            title = (paper.get("title") or "").lower()
            combined = abstract + " " + title

            hci_score, exclude_score, strong_count = compute_scores(combined)

            # Strict criteria:
            # 1. Must have at least 2 strong HCI signals
            # 2. HCI score must exceed exclude score by 3x
            # 3. Exclude score must be low
            if strong_count >= 2 and \
               hci_score > exclude_score * 3 and \
               exclude_score <= 2:
                fout.write(line)
                kept += 1
            else:
                removed += 1

            if total % 5_000_000 == 0:
                print(f"  Processed {total:,}... kept {kept:,} ({kept/total*100:.1f}%)")

    print(f"Total papers: {total:,}")
    print(f"Kept (HCI-relevant): {kept:,}")
    print(f"Removed (non-HCI): {removed:,}")
    print(f"Retention rate: {kept/total*100:.1f}%")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
