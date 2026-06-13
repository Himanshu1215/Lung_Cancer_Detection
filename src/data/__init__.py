LABEL_MAP: dict[str, int] = {"normal": 0, "nodule": 1}
CLASS_NAMES: list[str] = ["normal", "nodule"]
NUM_CLASSES: int = 2

# Malignancy-proxy task: Fleischner Society 6 mm threshold (diameter-based)
MALIG_LABEL_MAP: dict[str, int] = {"low_risk": 0, "high_risk": 1}
MALIG_CLASS_NAMES: list[str] = ["low_risk", "high_risk"]
