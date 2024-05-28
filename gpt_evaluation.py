import json
import os

import dotenv

import torch

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.evaluation.multi_modal import (
    MultiModalRelevancyEvaluator,
    MultiModalFaithfulnessEvaluator,
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal




# Always load the previous responses.
# response structure
# {
#     SYMBOL: {
#         "query_str": QUERY_STR,
#         "query": QUERY,
#         "texts": [
#            TEXT1(e.g: "To sign A in ASL: A closed fist, all ...),
#            TEXT2(e.g: "To sign B in ASL: B closed fist, all ...),
#            ...
#         ],
#         "imgs": [
#            IMG1(e.g: "./asl_dataset/images/K.jpg"),
#            IMG2(e.g: "./asl_dataset/images/L.jpg"),
#            ...
#         ],
#         "text_imgs": [
#            TEXT_IMG1(e.g: "./asl_dataset/images/K.jpg"),
#            TEXT_IMG2(e.g: "./asl_dataset/images/L.jpg"),
#            ...
#         ],
#         "img_response": IMG_RESPONSE,
#         "text_img_response": TEXT_IMG_RESPONSE,
#     },
#    SYMBOL2: {
#        ...
#    },
#    ...
#   "text_hitrate": 1.0,
#   "img_hitrate": 0.0,
#   "text_img_hitrate": 0.0,
#   "text_mrr": 1.0,
#   "img_mrr": 0.0,
#   "text_img_mrr": 0.0
# }

# API
OPENAI_API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY in .env file."
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
device = "cuda" if torch.cuda.is_available() else "cpu"

# exclude
exclude = [
    "text_hitrate",
    "img_hitrate",
    "text_img_hitrate",
    "text_mrr",
    "img_mrr",
    "text_img_mrr",
]

# functions
def evaluation_result_to_dict(result):
    dict_result = {}
    dict_result["passing"] = result.passing
    dict_result["feedback"] = result.feedback
    dict_result["score"] = result.score
    dict_result["contexts"] = result.contexts
    dict_result["response"] = result.response
    dict_result["query"] = result.query
    dict_result["invalid_result"] = result.invalid_result
    dict_result["invalid_reason"] = result.invalid_reason
    dict_result["pairwise_source"] = result.pairwise_source
    return dict_result
# load json file
response = {}
if os.path.exists("backup/response.json"):
    with open("backup/response.json", "r", encoding='utf-8') as f:
        response = json.load(f)

human_responses = {}
if os.path.exists("asl_dataset/human_responses.json"):
    with open("asl_dataset/human_responses.json", "r", encoding='utf-8') as f:
        human_responses = json.load(f)

judges = {}

judges["correctness"] = CorrectnessEvaluator(
    llm=OpenAI(temperature=0, model="gpt-4o"),
)

judges["relevancy"] = MultiModalRelevancyEvaluator(
    multi_modal_llm=OpenAIMultiModal(
        model="gpt-4o",
        max_new_tokens=300,
    )
)

judges["faithfulness"] = MultiModalFaithfulnessEvaluator(
    multi_modal_llm=OpenAIMultiModal(
        model="gpt-4o",
        max_new_tokens=300,
    )
)

gpt_eval = {}

for name, result in response.items():
    if name in exclude:
        continue
    print(f"Evaluating {name}...")

    reference_answer = human_responses[name]

    correctness_img_result = judges["correctness"].evaluate(
        query=result["query_str"],
        response=result["img_response"],
        reference_answer=human_responses[name],
    )

    relevancy_img_result = judges["relevancy"].evaluate(
        query=result["query_str"],
        response=result["img_response"],
        contexts=result["texts"],
        image_paths=result["imgs"],
    )

    faithfulness_img_result = judges["faithfulness"].evaluate(
        query=result["query_str"],
        response=result["img_response"],
        contexts=result["texts"],
        image_paths=result["imgs"],
    )
    correctness_img_result = evaluation_result_to_dict(correctness_img_result)
    relevancy_img_result = evaluation_result_to_dict(relevancy_img_result)
    faithfulness_img_result = evaluation_result_to_dict(faithfulness_img_result)

    correctness_text_img_result = judges["correctness"].evaluate(
        query=result["query_str"],
        response=result["text_img_response"],
        reference_answer=human_responses[name],
    )

    relevancy_text_img_result = judges["relevancy"].evaluate(
        query=result["query_str"],
        response=result["text_img_response"],
        contexts=result["texts"],
        image_paths=result["text_imgs"],
    )

    faithfulness_text_img_result = judges["faithfulness"].evaluate(
        query=result["query_str"],
        response=result["text_img_response"],
        contexts=result["texts"],
        image_paths=result["text_imgs"],
    )
    correctness_text_img_result = evaluation_result_to_dict(correctness_text_img_result)
    relevancy_text_img_result = evaluation_result_to_dict(relevancy_text_img_result)
    faithfulness_text_img_result = evaluation_result_to_dict(faithfulness_text_img_result)

    gpt_eval[name] = {
        "correctness_img_result": correctness_img_result,
        "relevancy_img_result": relevancy_img_result,
        "faithfulness_img_result": faithfulness_img_result,
        "correctness_text_img_result": correctness_text_img_result,
        "relevancy_text_img_result": relevancy_text_img_result,
        "faithfulness_text_img_result": faithfulness_text_img_result,
    }

# save gpt_eval to gpt_eval.json
with open("backup/gpt_eval.json", "w", encoding='utf-8') as f:
    json.dump(gpt_eval, f, indent=4)

