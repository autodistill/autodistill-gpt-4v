<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill GPT-4V Module

This repository contains the code supporting the GPT-4V base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[GPT-4V](https://openai.com/research/gpt-4v-system-card), developed by OpenAI, is a multi-modal language model. With GPT-4V, you can ask questions about images in natural language. The `autodistill-gpt4v` module enables you to classify images using GPT-4V.

This model uses the [gpt-4-vision-preview API](https://openai.com/blog/new-models-and-developer-products-announced-at-devday) announced by OpenAI on November 6th, 2023.

> [!NOTE]  
> Using this project will incur billing charges for API calls to the OpenAI GPT-4 Vision API.
> Refer to the [OpenAI pricing](https://openai.com/pricing) page for more information and to calculate your expected pricing. This package makes one API call per image you want to label.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [GPT-4V Autodistill documentation](https://autodistill.github.io/autodistill/base_models/gpt_4v/).

## Installation

To use GPT-4V with autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-gpt-4v
```

## Quickstart

```python
from autodistill_gpt_4v import GPT4V

# define an ontology to map class names to our GPT-4V prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GPT4V(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
base_model.label("./context_images", extension=".jpeg")
```

## License

This project is licensed under an [MIT license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!