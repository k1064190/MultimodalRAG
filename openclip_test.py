import json
import os

import chromadb
import qdrant_client
import dotenv
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import torch
import clip

from llama_index.core import PromptTemplate, ServiceContext
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex, VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, OpenAIEmbeddingFunction
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageDocument
from llama_index.core.schema import TextNode, ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)

# parameters
OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# functions
def plot_images(img_paths):
    images = 0
    fig, axs = plt.subplots(1, min(len(img_paths), 10), figsize=(30, 10))
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path).convert("RGB")
        axs[i].imshow(img)
        axs[i].axis("off")
        images += 1
        if images == 10:
            break
    plt.show()

def display_query_and_multimodal_response(query, response):
    print(f"Query: {query}")
    print("---------------------")
    print("Retreived multimodal response:")
    for r in response:
        print(r)
        if isinstance(r.node, TextNode):
            print(f"Text: {r.node.text}")
        elif isinstance(r.node, ImageNode):
            img = Image.open(r.node.metadata["file_path"]).convert("RGB")
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        print(f"Similarity: {r.score}")
        print("---------------------")

print(f"Your OpenAI API key is: {OPENAI_API_KEY}")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding_function = OpenCLIPEmbeddingFunction()
emb_model = HuggingFaceEmbedding(model_name='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', max_length =1024)
# create client and a new collection
image_loader = ImageLoader()
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

documents = SimpleDirectoryReader("./datasets/data_wiki/").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(image_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embedding_function, llm=None)

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    show_progress=True,
    is_image_to_text=False
)

retriever = index.as_retriever(image_similarity_top_k=5)

retrieval_results = retriever.retrieve("Vincent Van Gogh")

display_query_and_multimodal_response("Vincent Van Gogh", retrieval_results)


# context images
path_to_images = f"{dataset_path}/images"
img_paths = []
for img_path in os.listdir(path_to_images):
    img_paths.append(f"{path_to_images}/{img_path}")

plot_images(img_paths)


# context text
with open(f"{dataset_path}/asl_text_descriptions.json") as json_file:
    asl_text_descriptions = json.load(json_file)
text_documents = [
    Document(text=TEXT_TEMPLATE.format(symbol=k, desc=v))
    for k, v in asl_text_descriptions.items()
]

image_documents = SimpleDirectoryReader(path_to_images).load_data()
mixed_documents = text_documents + image_documents

node_parser = SentenceSplitter.from_defaults()
image_nodes = node_parser.get_nodes_from_documents(image_documents)
text_nodes = node_parser.get_nodes_from_documents(text_documents)

asl_index = MultiModalVectorStoreIndex.from_documents(
    mixed_documents, storage_context=storage_context, image_vector_store=image_store
)

# if not load_previously_generated_text_descriptions:
#     # define our lmm
#     openai_mm_llm = OpenAIMultiModal(
#         model="gpt-4o", max_new_tokens=300
#     )
#
#     # make a new copy since we want to store text in its attribute
#     image_with_text_documents = SimpleDirectoryReader(image_path).load_data()
#
#     # get text desc and save to text attr
#     for img_doc in tqdm.tqdm(image_with_text_documents):
#         response = openai_mm_llm.complete(
#             prompt="Describe the images as an alternative text",
#             image_documents=[img_doc],
#         )
#         img_doc.text = response.text
#
#     # save so don't have to incur expensive gpt-4v calls again
#     desc_jsonl = [
#         json.loads(img_doc.to_json()) for img_doc in image_with_text_documents
#     ]
#     with open(f"{dataset_path}/image_descriptions.json", "w") as f:
#         json.dump(desc_jsonl, f)
# else:
#     # load up previously saved image descriptions and documents
#     with open(f"{dataset_path}/image_descriptions.json") as f:
#         image_descriptions = json.load(f)
#
#     image_with_text_documents = [
#         ImageDocument.from_dict(el) for el in image_descriptions
#     ]


# openai_mm_llm = OpenAIMultiModal(
#     model="gpt-4o", max_new_tokens=300
# )


# # TEST
# symbol = "A"
# query_str = QUERY_STR_TEMPLATE.format(symbol=symbol)
# # response = rag_engines["rag_clip_gpt4o"].retrieve(query_str)
# response = retrieve_image(query_str)
#
# display_query_and_multimodal_response(query_str, response)

