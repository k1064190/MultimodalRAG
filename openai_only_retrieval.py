print("python started...")
import base64
import json
import os
print("Base imports done...")

import qdrant_client
import dotenv
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import torch
import clip
print("Additional imports done...")

from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader, Document, StorageContext
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import ImageDocument
from llama_index.core.schema import TextNode, ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
print("llama-index imports done...")

print("Starting...")

# parameters
OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
print(f"Your OpenAI API key is: {OPENAI_API_KEY}")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
device = "cuda" if torch.cuda.is_available() else "cpu"
load_previously_generated_text_descriptions = True
dataset_path = './Flickr8k_text'
output_path = 'flickr.json'

# Templates
# QUERY_STR_TEMPLATE = "How can I sign a {symbol}?."
QUERY_STR_FLICKR_TEMPLATE = "Give me a detailed description of the event where {caption}"
# QUERY_STR_FLICKR_TEMPLATE = "Give me a summary of the event where {caption} is happening."
TEXT_TEMPLATE = "To sign {symbol} in ASL: {desc}."
# QA_TEMPLATE_STR = (
#     "Images of hand gestures for ASL are provided.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "If the images provided cannot help in answering the query\n"
#     "then respond that you are unable to answer the query. Otherwise,\n"
#     "using only the context provided, and not prior knowledge,\n"
#     "provide an answer to the query."
#     "Query: {query_str}\n"
#     "Answer: "
# )
# QA_TEMPLATE = PromptTemplate(QA_TEMPLATE_STR)
QA_FLICKR_TEMPLATE_STR = (
    "Images of events are provided.\n"
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
QA_TEMPLATE = PromptTemplate(QA_FLICKR_TEMPLATE_STR)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# local qdrant client
qdrant_client = qdrant_client.QdrantClient(location=":memory:")
# text_store = QdrantVectorStore(
#     client=qdrant_client, collection_name="text_collection"
# )
image_store = QdrantVectorStore(
    client=qdrant_client, collection_name="image_collection"
)
img_text_store = QdrantVectorStore(
    client=qdrant_client, collection_name="text_collection"
)
text_embedding = OpenAIEmbedding(model="text-embedding-3-large")

print("available models: ", clip.available_models())
model, preprocess = clip.load("ViT-L/14", device=device)

# context images
path_to_images = f"Flicker8k_Dataset"
path_to_text = f"Flickr8k_text"
# img_paths = []
# for img_path in os.listdir(path_to_images):
#     img_paths.append(f"{path_to_images}/{img_path}")

with open(f"{path_to_text}/Flickr_8k.testImages.txt") as f:
    test_imgs = f.read().splitlines()
    test_imgs = list(map(lambda x: f"{path_to_images}/{x}", test_imgs))
    img_paths = test_imgs

# image to caption
captions = {}
text_documents = []
with open(f"{path_to_text}/Flickr8k.token.txt") as f:
    lines = f.readlines()
    for line in lines:
        img, caption = line.split("\t")
        img = img.split("#")[0]
        caption = caption.strip()
        if img in captions:
            captions[img].append(caption)
        else:
            captions[img] = [caption]

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


# # context text
# with open(f"{dataset_path}/asl_text_descriptions.json") as json_file:
#     asl_text_descriptions = json.load(json_file)
# text_documents = [
#     Document(text=TEXT_TEMPLATE.format(symbol=k, desc=v))
#     for k, v in asl_text_descriptions.items()
# ]
#
# text_storage = StorageContext.from_defaults(vector_store=text_store)
# text_index = VectorStoreIndex.from_documents(
#     text_documents,
#     storage_context=text_storage,
#     embed_model=text_embedding
# )
# # text_retreiver
# text_retriever = text_index.as_retriever(similarity_top_k=3)

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
    storage_context=image_storage,
)

if not load_previously_generated_text_descriptions:
    # define our lmm
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o", max_new_tokens=300
    )

    # img_paths to ImageDocument
    image_to_text_document = [
        ImageDocument(
            text="",
            metadata={"file_path": img_path}
        )
        for img_path in img_paths
    ]

    # get text desc and save to text attr
    with open(f"{dataset_path}/image_descriptions.json", "w") as f:
        for img_doc in tqdm.tqdm(image_to_text_document):
            # catch the error
            try:
                img_response = openai_mm_llm.complete(
                    prompt="Describe the images as a detailed alternative text",
                    image_documents=[img_doc],
                )
                img_doc.text = img_response.text
            except Exception as e:
                print(f"Error: {e} on {img_doc.metadata['file_path']}")
                # continue the loop
                continue

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

print(f"Number of image descriptions: {len(image_to_text_document)}")

# ImageDocument to TextDocument
img_text_document = []
img_text_paths = []
for img_doc in image_to_text_document:
    new_text_doc = Document(text=img_doc.text, metadata={"file_path": img_doc.metadata["file_path"]} )
    img_text_document.append(new_text_doc)
    img_text_paths.append(img_doc.metadata["file_path"])

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
        elif isinstance(node, ImageNode):
            img = Image.open(node.metadata["file_path"]).convert("RGB")
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        print(f"Similarity: {sim}")
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
text_hit_num = 0
img_hit_num = 0
text_img_hit_num = 0
text_mrr = 0
img_mrr = 0
text_img_mrr = 0
results = {}
for img_path in img_text_paths:
    this_img_hit_num = 0
    this_img_mrr = 0
    this_text_img_hit_num = 0
    this_text_img_mrr = 0
    img_name = img_path.split("/")[-1]
    caps = captions[img_name]
    query_strs = []
    for cap in caps:
        query_str = QUERY_STR_FLICKR_TEMPLATE.format(caption=cap)
        query_strs.append(query_str)
        response_img = retrieve_image(query_str)
        response_text_img = img_text_retriever.retrieve(query_str)
        # response_texts = text_retriever.retrieve(query_str)

        # texts = list(map(lambda x: x.node.text, response_texts))
        text_imgs = list(map(lambda x: x.node.metadata["file_path"], response_text_img))
        imgs = list(map(lambda node: node.metadata["file_path"], response_img.nodes))

        # # calculate hitrate and mrr
        # for i, t in enumerate(texts):
        #     # t = "To sign A in ASL: A is formed by making a fist with your thumb extended and placing it on your chin."
        #     if t.split(":")[0].strip().lower() == f"To sign {s} in ASL".lower():
        #         text_hit_num += 1
        #         text_mrr += 1/(i+1)
        #         break

        for i, img in enumerate(imgs):
            # img = "asl_dataset/images/A.jpg"
            if img.split("/")[-1].split(".")[0].lower() == img_name.split(".")[0].lower():
                this_img_hit_num += 1
                this_img_mrr += 1 / (i + 1)
                break
        for i, img in enumerate(text_imgs):
            # img = "asl_dataset/images/A.jpg"
            if img.split("/")[-1].split(".")[0].lower() == img_name.split(".")[0].lower():
                this_text_img_hit_num += 1
                this_text_img_mrr += 1 / (i + 1)
                break
    this_img_hit_num /= len(caps)
    this_img_mrr /= len(caps)
    this_text_img_hit_num /= len(caps)
    this_text_img_mrr /= len(caps)

    img_hit_num += this_img_hit_num
    img_mrr += this_img_mrr
    text_img_hit_num += this_text_img_hit_num
    text_img_mrr += this_text_img_mrr

    results[img_path] = {
        "query_str": query_strs,
        # "query": query,
        # "texts": texts,
        "img_hitnum": this_img_hit_num,
        "text_img_hitnum": this_text_img_hit_num,
        "img_mrr": this_img_mrr,
        "text_img_mrr": this_text_img_mrr,
    }

# results["text_hitrate"] = text_hit_num / len(img_paths)
results["img_hitrate"] = img_hit_num / len(img_text_paths)
results["text_img_hitrate"] = text_img_hit_num / len(img_text_paths)
# results["text_mrr"] = text_mrr / len(img_text_paths)
results["img_mrr"] = img_mrr / len(img_text_paths)
results["text_img_mrr"] = text_img_mrr / len(img_text_paths)

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)
