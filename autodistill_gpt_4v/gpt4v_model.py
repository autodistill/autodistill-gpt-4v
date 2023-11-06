import base64
import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from openai import OpenAI

HOME = os.path.expanduser("~")
SUPPORTED_PREDICTION_TYPES = ["few-shot", "zero-shot"]


def processed_predictions(response, ontology):
    predictions = response.choices[0].text.split("\n")

    xyxys = [line.split(" ") for line in predictions]
    classes = [xyxys[-1] for _ in range(len(xyxys))]
    class_ids = [ontology.prompts().index(item) for item in classes]

    xyxy_coordinates = [obj[:-1] for obj in xyxys]

    return sv.Detections(
        xyxy=np.array(xyxy_coordinates),
        confidence=np.array([1] * len(xyxys)),
        class_id=class_ids,
    )


@dataclass
class GPT4V(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.ontology = ontology
        pass

    def few_shot_predict(self, images, coordinates):
        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"I would like to identify the {', '.join(self.ontology.prompts())} in images. Here are some examples of the {', '.join(self.ontology.prompts())} in images. In order, they are: {', '.join(self.ontology.prompts())}. In order, their bounding box coordinates are {' | '.join(', '.join(coordinates))}.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Return the coordinates of the {', '.join(self.ontology.prompts())} in the following image. Return each prediction on a new line as x0, y0, x1, y1, class bounding box coordinates.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open(input, "rb").read()).decode(
                                "utf-8"
                            ),
                        },
                    },
                ],
            },
        ]

        for image in images:
            payload[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,"
                        + base64.b64encode(open(image, "rb").read()).decode("utf-8"),
                    },
                }
            )

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=payload,
            max_tokens=300,
        )

        class_ids = self.ontology.prompts().index(response.choices[0].text)

        return sv.Classifications(
            xyxy=np.array(class_ids),
            confidence=np.array([1]),
        )

    def predict_classify(self, input, classes) -> sv.Classifications:
        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"What is in the image? Return the class of the object in the image. Here are the classes: {', '.join(classes)}. You can only return one class from that list.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open(input, "rb").read()).decode(
                                "utf-8"
                            ),
                        },
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=payload,
            max_tokens=300,
        )

        return processed_predictions(response, self.ontology)

    def predict(self, input: str) -> sv.Detections:
        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Return the coordinates of the {', '.join(self.ontology.prompts())} in the following image. Return each prediction on a new line as x0, y0, x1, y1, class bounding box coordinates.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open(input, "rb").read()).decode(
                                "utf-8"
                            ),
                        },
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=payload,
            max_tokens=300,
        )

        return processed_predictions(response, self.ontology)
