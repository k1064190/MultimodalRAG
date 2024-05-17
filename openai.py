from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
import dotenv
import tqdm

from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument

# parameters
OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
load_previously_generated_text_descriptions = True
dataset_path = "./asl_dataset"

# Templates
QUERY_STR_TEMPLATE = "How can I sign a {symbol}?."
TEXT_TEMPLATE = "To sign {symbol} in ASL: {desc}."

# context images
image_path = f"{dataset_path}/images"
image_documents = SimpleDirectoryReader(image_path).load_data()

# context text
with open(f"{dataset_path}/asl_text_descriptions.json") as json_file:
    asl_text_descriptions = json.load(json_file)
text_documents = [
    Document(text=TEXT_TEMPLATE.format(symbol=k, desc=v))
    for k, v in asl_text_descriptions.items()
]

node_parser = SentenceSplitter.from_defaults()
image_nodes = node_parser.get_nodes_from_documents(image_documents)
text_nodes = node_parser.get_nodes_from_documents(text_documents)

asl_index = MultiModalVectorStoreIndex(image_nodes + text_nodes)

if not load_previously_generated_text_descriptions:
    # define our lmm
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", max_new_tokens=300
    )

    # make a new copy since we want to store text in its attribute
    image_with_text_documents = SimpleDirectoryReader(image_path).load_data()

    # get text desc and save to text attr
    for img_doc in tqdm.tqdm(image_with_text_documents):
        response = openai_mm_llm.complete(
            prompt="Describe the images as an alternative text",
            image_documents=[img_doc],
        )
        img_doc.text = response.text

    # save so don't have to incur expensive gpt-4v calls again
    desc_jsonl = [
        json.loads(img_doc.to_json()) for img_doc in image_with_text_documents
    ]
    with open("image_descriptions.json", "w") as f:
        json.dump(desc_jsonl, f)
else:
    # load up previously saved image descriptions and documents
    with open("asl_data/image_descriptions.json") as f:
        image_descriptions = json.load(f)

    image_with_text_documents = [
        ImageDocument.from_dict(el) for el in image_descriptions
    ]

# parse into nodes
image_with_text_nodes = node_parser.get_nodes_from_documents(
    image_with_text_documents
)
