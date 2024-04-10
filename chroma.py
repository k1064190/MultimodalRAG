import base64

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.utils import truncate_text
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from IPython.display import Markdown, display
from llama_index.llms.llama_cpp.base import LlamaCPP, Llama
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
import chromadb
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from chromadb.utils.data_loaders import ImageLoader
from llama_index.core.schema import ImageNode, MetadataMode

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

import matplotlib.pyplot as plt

prompt = "Explain me about the painting of Van Gogh."

# set defalut text and image embedding functions
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

# load documents
documents = SimpleDirectoryReader("datasets/data_wiki/").load_data()

# create client and a new collection
chroma_client = chromadb.PersistentClient(path='chroma_db')
chroma_collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model='local',
    # embed_model=embedding_function
)

llama_url = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
llava_url = "https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K.gguf"

# llm = LlamaCPP(
#     model_url=llava_url,
#     model_path=None,
#     temperature=0.1,
#     max_new_tokens=256,
#     context_window=4096,
#     generate_kwargs={},
#     model_kwargs={"n_gpu_layers": -1},
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
# )

# set up ChromaVectorStore and load in data
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# service_context = ServiceContext.from_defaults(chunk_size=1024, embed_model="local", llm=llm)
# Settings.llm = llm
# Settings.embed_model = "local"
# storage_context = StorageContext.from_defaults(
#     vector_store=vector_store,
# )
# index = VectorStoreIndex.from_documents(
#     documents,
#     storage_context=storage_context,
#     service_context=service_context,
# )

retriever = index.as_retriever(similarity_top_k=10)
retrieval_results = retriever.retrieve(prompt)

# image_results = []
# MAX_RES = 5
# cnt = 0
# for r in retrieval_results:
#     if isinstance(r.node, ImageNode):
#         image_results.append(r.node.metadata["file_path"])
#     else:
#         if cnt < MAX_RES:
#             display_source_node(r)
#             metadata_mode = MetadataMode.NONE
#             source_length = 100
#             source_text_fmt = truncate_text(
#                 r.node.get_content(metadata_mode=metadata_mode).strip(), source_length
#             )
#             text_md = (
#                 f"**Node ID:** {r.node.node_id}",
#                 f"**Similarity:** {r.score}",
#                 f"**Text:** {source_text_fmt}",
#             )
#             for t in text_md:
#                 print(t)
#         cnt += 1

# display_image_uris(image_results, (3, 3), top_k=2)
# plt.show()
#
# query = index.as_query_engine(llm=llm)

chat_handler = Llava15ChatHandler(
    clip_model_path="datasets/llava-v1.5-7b-mmproj-Q4_0.gguf"
)
llm = Llama(
    model_path="datasets/llava-v1.5-7b-Q4_K.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,
    logits_all=True,
)

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

# transform retrieval results to chat format
# e.g) {"type": "image_url", "image_url": {"url": data_uri}}, {"type": "text", "text" : prompt}
chat = []
source_length = 100
metadata_mode = MetadataMode.NONE
MAX_RES = 5
cnt = 0
for r in retrieval_results:
    if isinstance(r.node, ImageNode):
        data_uri = image_to_base64_data_uri(r.node.metadata["file_path"])
        chat.append({"type": "image_url", "image_url": {"url": data_uri}})
    else:
        if cnt < MAX_RES:
            chat.append({"type": "text", "text":
                truncate_text(r.node.get_content(metadata_mode=metadata_mode).strip(), source_length)})
        cnt += 1
print(chat)

response = llm.create_chat_completion(
    messages=[
        {"role" : "system", "content" : "You are an assistant who helps users find information with help of multimodal data(image, text)."},
        {
            "role": "user",
            "content": [
                *chat,
                {"type": "text", "text" : prompt},
            ]
        }
    ],
)

print(response)
print(response.choices[0].message.content)