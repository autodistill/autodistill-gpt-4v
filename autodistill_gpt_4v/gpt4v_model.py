import base64
import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from openai import OpenAI

HOME = os.path.expanduser("~")

# def processed_predictions(response, ontology, height, width):
#     predictions = response.choices[0].message.content.split("\n")

#     if "None" in predictions:
#         return sv.Detections.empty()

#     xyxys = predictions[0].split(", ")
#     class_ids = [0]

#     xyxys = [float(x) for x in xyxys]

#     # normalize from 0-1 to image coordinates
#     # coordinates are x_min, y_min, x_max, y_max
#     xyxys[0] = xyxys[0] * height
#     xyxys[1] = xyxys[1] * width
#     xyxys[2] = xyxys[2] * height
#     xyxys[3] = xyxys[3] * width

#     xyxys = [int(xyxy) for xyxy in xyxys]

#     # xyxy_coordinates = [obj[:-1] for obj in xyxys]

#     return sv.Detections(
#         xyxy=np.array([xyxys]),
#         confidence=np.array([1]),
#         class_id=np.array(class_ids),
#     )


@dataclass
class GPT4V(DetectionBaseModel):
    ontology: CaptionOntology
    api_key: str = None

    def __init__(self, ontology: CaptionOntology, api_key: str = None):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.ontology = ontology
        pass

    def predict(self, input, classes) -> sv.Classifications:
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
                        }
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=payload,
            max_tokens=300,
        )

        class_ids = self.ontology.prompts().index(response.choices[0].message.content)

        return sv.Classifications(
            class_id=np.array([class_ids]),
            confidence=np.array([1]),
        )

    # def predict(self, input: str) -> sv.Detections:
        # payload = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": f"Can you tell me the location of the dog on the image. Share the x_min, y_min, x_max and y_max in 0-1 normalized space. Only return the numbers, nothing else."
        #             },
        #             {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": f"data:image/jpeg;base64,"
        #                     + base64.b64encode(open(input, "rb").read()).decode(
        #                         "utf-8"
        #                     ),
        #                 },
        #             },
        #         ],
        #     }
        # ]

        # print(payload[0]["content"][0])

        # from PIL import Image

        # height, width = Image.open(input).size

        # # exit()

        # response = self.client.chat.completions.create(
        #     model="gpt-4-vision-preview",
        #     messages=payload,
        #     max_tokens=300,
        # )

        # print(response)

        # return processed_predictions(response, self.ontology, height, width)
