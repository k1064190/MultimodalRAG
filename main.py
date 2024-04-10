import os

import torch

from llama_cpp import Llama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, ServiceContext
from llama_index.llms.llama_cpp.base import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.core.node_parser import SentenceWindowNodeParser

# llm = Llama.from_pretrained(
#     repo_id="TheBloke/Llama-2-7B-GGUF",
#     filename="llama-2-7b.Q4_K_M.gguf",
# )

llm = LlamaCPP(
    model_url="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf",
    model_path="llama-2-7b.Q4_K_M.gguf",
    temperature=0.1,
    max_new_tokens=256,
    context_window=4096,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

documents = SimpleDirectoryReader(
    input_files=["./documents/A survey of large language models.pdf"]
).load_data()

documents = Document(
    text = "\n\n".join([doc.text for doc in documents])
)