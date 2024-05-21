import base64
import json
import os

import chromadb
import qdrant_client
import dotenv
import matplotlib.pyplot as plt
import requests
import tqdm
from PIL import Image
import torch
import clip

from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex, VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from chromadb.utils.data_loaders import ImageLoader
from llama_index.core.schema import ImageDocument
from llama_index.core.schema import TextNode, ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

# parameters
OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
device = "cuda" if torch.cuda.is_available() else "cpu"
load_previously_generated_text_descriptions = True
dataset_path = "./asl_dataset"
output_path = 'output.txt'

out_file = open(output_path, "w")

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

print(f"Your OpenAI API key is: {OPENAI_API_KEY}")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

qdrant_client = qdrant_client.QdrantClient(location=":memory:")
text_store = QdrantVectorStore(
    client=qdrant_client, collection_name="text_collection"
)
image_store = QdrantVectorStore(
    client=qdrant_client, collection_name="image_collection"
)
img_text_store = QdrantVectorStore(
    client=qdrant_client, collection_name="text_collection"
)
text_embedding = OpenAIEmbedding(model="text-embedding-3-large")

print("available models: ", clip.available_models())
model, preprocess = clip.load("ViT-B/32", device=device)

# context images
path_to_images = f"{dataset_path}/images"
img_paths = []
for img_path in os.listdir(path_to_images):
    img_paths.append(f"{path_to_images}/{img_path}")

img_emb_dict = {}
with torch.no_grad():
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        if os.path.isfile(img_path):
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            img_emb = model.encode_image(img)
            # print(img_emb)
            img_emb_dict[filename] = img_emb.cpu().numpy().tolist()[0]
print(len(img_emb_dict))

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

# plot_images(img_paths)


# context text
with open(f"{dataset_path}/asl_text_descriptions.json") as json_file:
    asl_text_descriptions = json.load(json_file)
text_documents = [
    Document(text=TEXT_TEMPLATE.format(symbol=k, desc=v))
    for k, v in asl_text_descriptions.items()
]

text_storage = StorageContext.from_defaults(vector_store=text_store)
text_index = VectorStoreIndex.from_documents(
    text_documents,
    storage_context=text_storage,
    embed_model=text_embedding
)
# text_retreiver
text_retriever = text_index.as_retriever(similarity_top_k=3)

image_documents = []
for filename, img_emb in img_emb_dict.items():
    new_img_doc = ImageDocument(
        text=filename,
        metadata={"file_path": f"{path_to_images}/{filename}"},
    )
    new_img_doc.embedding = img_emb
    image_documents.append(new_img_doc)

image_storage = StorageContext.from_defaults(vector_store=image_store)
image_index = VectorStoreIndex.from_documents(
    image_documents,
    storage_context=image_storage
)

if not load_previously_generated_text_descriptions:
    # define our lmm
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o", max_new_tokens=300
    )

    # make a new copy since we want to store text in its attribute
    image_to_text_document = SimpleDirectoryReader(path_to_images).load_data()

    # get text desc and save to text attr
    for img_doc in tqdm.tqdm(image_to_text_document):
        response = openai_mm_llm.complete(
            prompt="Describe the images as an alternative text",
            image_documents=[img_doc],
        )
        img_doc.text = response.text

    # save so don't have to incur expensive gpt-4v calls again
    desc_jsonl = [
        json.loads(img_doc.to_json()) for img_doc in image_to_text_document
    ]
    with open(f"{dataset_path}/image_descriptions.json", "w") as f:
        json.dump(desc_jsonl, f)
else:
    # load up previously saved image descriptions and documents
    with open(f"{dataset_path}/image_descriptions.json") as f:
        image_descriptions = json.load(f)

    image_to_text_document = [
        ImageDocument.from_dict(el) for el in image_descriptions
    ]

# ImageDocument to TextDocument
img_text_document = []
for img_doc in image_to_text_document:
    new_text_doc = Document(text=img_doc.text, metadata={"file_path": img_doc.metadata["file_path"]} )
    img_text_document.append(new_text_doc)

node_parser = SentenceSplitter.from_defaults()
image_text_nodes = node_parser.get_nodes_from_documents(image_to_text_document)

img_text_storage = StorageContext.from_defaults(vector_store=img_text_store)
img_text_index = VectorStoreIndex(
    nodes=image_text_nodes,
    storage_context=img_text_storage,
    embed_model=text_embedding
)
# text_retreiver
img_text_retriever = img_text_index.as_retriever(similarity_top_k=3)

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

def display_query_and_multimodal_response2(query, response):
    print(f"Query: {query}")
    print("---------------------")
    print("Retreived multimodal response:")
    nodes = response.nodes
    sims = response.similarities
    for r in zip(nodes, sims):
        node, sim = r
        if isinstance(node, TextNode):
            print(f"Text: {node.text}")
            print(f"Imgpath: {node.metadata['file_path']}")
            out_file.write(f"Text: {node.text}\n")
            out_file.write(f"Imgpath: {node.metadata['file_path']}\n")
        elif isinstance(node, ImageNode):
            img = Image.open(node.metadata["file_path"]).convert("RGB")
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        print(f"Similarity: {sim}")
        out_file.write(f"Similarity: {sim}\n")
        print("---------------------")


def retrieve_image(query):
    tokens = clip.tokenize(query).to(device)
    query_emb = model.encode_text(tokens).tolist()[0]
    image_query = VectorStoreQuery(
        query_emb, similarity_top_k=3
    )
    response = image_store.query(image_query)

    return response

# TEST
# symbol = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
#           "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
#           "U", "V", "W", "X", "Y", "Z"]
symbol = ["A"]

text_hit_num = 0
img_hit_num = 0
text_img_hit_num = 0
for s in symbol:
    query_str = QUERY_STR_TEMPLATE.format(symbol=s)
    response_img = retrieve_image(query_str)
    response_text_img = img_text_retriever.retrieve(query_str)
    response_texts = text_retriever.retrieve(query_str)
    text = ""
    for r in response_texts:
        if isinstance(r.node, TextNode):
            text += f"{r.node.text}\n"

    print(response_text_img)
    retrieved_text_img = response_text_img[0].node.metadata["file_path"]
    print("IMGS: ", retrieved_text_img)
    print("TEXT: ", text)

    # print("Using openai clip embedding")
    # display_query_and_multimodal_response2(query_str, response_img)
    # print("\n")
    # retrieved_img = response_img.nodes[0]
    # img_url = retrieved_img.metadata["file_path"]
    # base64_img = encode_image(img_url)

    # query = QA_TEMPLATE_STR.format(context_str=text,query_str=query_str)

    # print("Generated query")
    # print(query)
    # out_file.write(query)
    #
    # client = OpenAI()
    #
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "text",
    #                     "text": query
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{base64_img}"
    #                     }
    #                 }
    #             ]
    #         }
    #     ],
    # )
    #
    # print(f"Answer: {response.choices[0].message.content}")
    # out_file.write(f"Answer: {response.choices[0].message.content}\n\n")

out_file.close()

# img_document = ImageDocument(
#     text="A",
#     metadata={"file_path": retrieved_img.metadata["file_path"]},
#     embedding=retrieved_img.embedding
# )

# Using the retrieved image, we can now query the multimodal rag llm

