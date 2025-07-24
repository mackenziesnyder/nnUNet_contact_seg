map_acronyms = {
        "A": "Anterior",
        "L": "Left",
        "R": "Right",
        "C": "Center",
        "Ce": "Central",
        "P": "Posterior",
        "M": "Medial",
        "I": "Interior",
        "S": "Superior",
        "Fr": "Frontal",
        "Inf": "Infra",
        "Int": "Intra",
        "Ln": "Line",
        "Ma": "Margin",
        "Me": "Medial",
        "Mi": "Middle",
        "Mo": "Motor",
        "Or": "Orthogonal",
        "Parc": "Paracentral",
        "Par": "Para",
        "Pr": "Pre",
        "Sa": "Sagittal",
        "Sp": "Supplementary",
        "Sup": "Supra",
        "Sc": "Sensory",
        "Mc": "Motor cortex",
        "Ps": "Parasagittal",
        "Am": "Amygdala",
        "An": "Angular",
        "Ca": "Calcarine",
        "Br": "Broca",
        "Cc": "Corpus Callosum",
        "Cg": "Cingulate",
        "Cl": "Cleft",
        "Cu": "Cuneus",
        "Cx": "Convex",
        "Dy": "Dysplasia",
        "Ec": "Entorhinal Cortex",
        "Fa": "Face",
        "Gy": "Gyrus",
        "Ha": "Hand",
        "Hc": "Hippocampus",
        "Hs": "Heschel’s",
        "Ht": "heterotopia",
        "In": "Insula",
        "Lg": "Lingula",
        "Lo": "Lobe",
        "Ls": "Lesion",
        "O": "Orbital",
        "Oc": "Occipital",
        "Op": "Operculum",
        "Po": "Pole",
        "Pa": "Parietal",
        "Ps": "Parasagittal",
        "Ro": "Rolandic",
        "Re": "Resection",
        "Rs": "Rostrum",
        "Sl": "Splenium",
        "SMA": "Supplementary Motor Area",
        "Te": "Temporal",
        "Unc": "Uncus",
        "We": "Wernicke"
    }

def convert_acronym_to_words(label):
    label = label.strip()
    words = []
    i = 0
    while i < len(label):
        matched = False
        for j in range(3, 0, -1):
            part = label[i:i+j]
            if part in map_acronyms:
                words.append(map_acronyms[part])
                i += j
                matched = True
                break
        if not matched:
            raise ValueError(f"Unknown acronym sequence starting at: '{label[i:]}'")
    return " ".join(words)




