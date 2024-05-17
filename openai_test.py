import json
import os

import dotenv
import matplotlib.pyplot as plt
import tqdm
from PIL import Image

from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageDocument
from llama_index.core.schema import TextNode, ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# parameters
OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
load_previously_generated_text_descriptions = True
dataset_path = "./asl_dataset"

# Templates
QUERY_STR_TEMPLATE = "How can I sign a {symbol}?."
TEXT_TEMPLATE = "To sign {symbol} in ASL: {desc}."
QA_TEMPLATE_STR = (
    "Images of hand gestures for ASL are provided.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "If the images provided cannot help in answering the query\n"
    "then respond that you are unable to answer the query. Otherwise,\n"
    "using only the context provided, and not prior knowledge,\n"
    "provide an answer to the query."
    "Query: {query_str}\n"
    "Answer: "
)
QA_TEMPLATE = PromptTemplate(QA_TEMPLATE_STR)

print(f"Your OpenAI API key is: {OPENAI_API_KEY}")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
    with open(f"{dataset_path}/image_descriptions.json", "w") as f:
        json.dump(desc_jsonl, f)
else:
    # load up previously saved image descriptions and documents
    with open(f"{dataset_path}/image_descriptions.json") as f:
        image_descriptions = json.load(f)

    image_with_text_documents = [
        ImageDocument.from_dict(el) for el in image_descriptions
    ]

# parse into nodes
image_with_text_nodes = node_parser.get_nodes_from_documents(
    image_with_text_documents
)

asl_text_desc_index = MultiModalVectorStoreIndex(
    nodes=image_with_text_nodes + text_nodes, is_image_to_text=True
)

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", max_new_tokens=300
)

# define our RAG query engines
rag_engines = {
    "mm_clip_gpt4o": asl_index.as_query_engine(
        multi_modal_llm=openai_mm_llm, text_qa_template=QA_TEMPLATE
    ),
    # "mm_clip_llava": asl_index.as_query_engine(
    #     multi_modal_llm=llava_mm_llm,
    #     text_qa_template=qa_tmpl,
    # ),
    "mm_text_desc_gpt4o": asl_text_desc_index.as_query_engine(
        multi_modal_llm=openai_mm_llm, text_qa_template=QA_TEMPLATE
    ),
    # "mm_text_desc_llava": asl_text_desc_index.as_query_engine(
    #     multi_modal_llm=llava_mm_llm, text_qa_template=qa_tmpl
    # ),
}

def display_query_and_multimodal_response(query, response):
    print(f"Query: {query}")
    print("---------------------")
    print("Retreived multimodal response:")
    for r in response:
        if isinstance(r.node, TextNode):
            print(f"Text: {r.node.text}")
        elif isinstance(r.node, ImageNode):
            img = Image.open(r.node.image_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        print(f"Similarity: {r.score}")
        print("---------------------")

# TEST
symbol = "A"
query_str = QUERY_STR_TEMPLATE.format(symbol=symbol)
response = rag_engines["mm_clip_gpt4o"].query(query_str)

display_query_and_multimodal_response(query_str, response)
