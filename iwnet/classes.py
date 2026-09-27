"""Class metadata: display names, plain-language descriptions and guidance.

Kept in one place so the CLI, the API and the UI all describe the classes
identically. The misspelling ``Irrelavant`` is the project's intentional
rejection-class name and is preserved everywhere for compatibility.
"""

from __future__ import annotations

from iwnet.config import FINAL_CLASSES

__all__ = ["CLASS_INFO", "is_rejection_class", "describe_class", "class_order"]

#: ``severity`` is informational only - it is NOT a diagnosis and the model
#: does not estimate it.
CLASS_INFO: dict[str, dict[str, str]] = {
    "BacterialSpot": {
        "label": "Bacterial Spot",
        "short": "Bacterial leaf spot",
        "description": (
            "Small, dark, water-soaked or necrotic spots on the leaf surface, often "
            "surrounded by a yellow halo. Spots may coalesce into larger dead patches."
        ),
        "note": "Bacterial infections are generally managed with copper-based protectants and "
                "by reducing leaf wetness; a laboratory diagnosis is needed to confirm the "
                "causative species.",
    },
    "Black_Rot": {
        "label": "Black Rot",
        "short": "Black rot",
        "description": (
            "Circular tan lesions with a dark brown margin and a ring of black pycnidia, "
            "often starting near the leaf margin."
        ),
        "note": "Typically favoured by warm, humid conditions and damaged fruit. Removing "
                "mummified tissue and improving canopy airflow are common cultural controls.",
    },
    "DownyMildew": {
        "label": "Downy Mildew",
        "short": "Downy mildew",
        "description": (
            "Angular, oil-like yellow lesions bounded by leaf veins on the upper surface, "
            "with a white downy fungal growth on the underside in humid conditions."
        ),
        "note": "A major disease of grapevine. Needs early detection; protective fungicide "
                "programmes are typically region- and season-specific.",
    },
    "Esca": {
        "label": "Esca (Black Measles)",
        "short": "Esca / black measles",
        "description": (
            "Interveinal chlorosis and necrosis producing a tiger-stripe pattern, often with "
            "small dark spots on the remaining green tissue."
        ),
        "note": "A trunk disease associated with several fungi. Foliar symptoms typically "
                "appear years after infection, so vine age and history matter.",
    },
    "Healthy": {
        "label": "Healthy",
        "short": "Healthy leaf",
        "description": "No visible symptoms of the diseases represented in this dataset.",
        "note": "Absence of detectable symptoms is not a guarantee of health - very early or "
                "very mild infections may not be visible in a photograph.",
    },
    "Irrelavant": {
        "label": "Irrelavant (not a grape leaf)",
        "short": "Not a grape leaf",
        "description": (
            "The image does not show a grapevine leaf. This is the model's rejection class: "
            "it exists so the system can decline to give a disease verdict for unrelated "
            "photographs."
        ),
        "note": "Nothing to diagnose. If this really is a grape leaf, the photo is probably "
                "too unclear, too zoomed-out, or shows a symptom pattern not covered by "
                "training data.",
    },
    "PowderyMildew": {
        "label": "Powdery Mildew",
        "short": "Powdery mildew",
        "description": (
            "White-grey powdery fungal growth on leaves, berries or shoots, often with "
            "brown speckling or slight curling on affected tissue."
        ),
        "note": "Unlike downy mildew it does not require free water on the leaf, so it can "
                "spread in dry, warm conditions.",
    },
}

#: The rejection class. Exact spelling is load-bearing.
REJECTION_CLASS = "Irrelavant"


def is_rejection_class(name: str) -> bool:
    return name == REJECTION_CLASS


def describe_class(name: str) -> dict[str, str]:
    info = CLASS_INFO.get(name)
    if info is None:
        return {
            "label": name,
            "short": name,
            "description": "No description available for this class.",
            "note": "",
        }
    return {**info, "name": name, "is_rejection": is_rejection_class(name)}


def class_order() -> list[dict[str, str]]:
    """All 7 classes with their metadata, in canonical order."""
    return [describe_class(name) for name in FINAL_CLASSES]
