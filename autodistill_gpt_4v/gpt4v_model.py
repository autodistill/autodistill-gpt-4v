import os
from dataclasses import dataclass

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")

@dataclass
class GPT4V(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        pass

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        pass