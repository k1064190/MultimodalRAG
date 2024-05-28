import json

# {
#   "A": {
#     "correctness_img_result": {
#       "passing": false,
#       "feedback": "The generated answer is somewhat relevant but contains mistakes. The user query is asking how to sign the letter \"A\" in ASL, but the generated answer incorrectly references images and provides an unnecessary explanation about the absence of images. The correct part of the answer, which describes how to sign the letter \"A,\" is buried within irrelevant information.",
#       "score": 2.0,
#       "contexts": null,
#       "response": "The images provided do not show the hand gesture for the letter \"A\" in American Sign Language (ASL) based on the description given. According to the description, to sign the letter \"A\" in ASL, you would make a closed fist with all fingers folded against the palm, and the thumb is straight, alongside the index finger. \n\nTherefore, I am unable to provide the image of the correct sign for \"A\" in ASL from the provided images.",
#       "query": "How can I sign a A?.",
#       "invalid_result": false,
#       "invalid_reason": null,
#       "pairwise_source": null
#     },
#     "relevancy_img_result": ...,
#     "faithfulness_img_result": ...,
#     "correctness_text_img_result": ...,
#     "relevancy_text_img_result": ...,
#     "faithfulness_text_img_result
#   },
#   "B": ...,
# }


with open('backup/gpt_eval.json', 'r') as json_file:
    evaluations = json.load(json_file)

    # Get the mean score for each metric
    correctness_scores = []
    relevancy_scores = []
    faithfulness_scores = []
    correctness_text_img_scores = []
    relevancy_text_img_scores = []
    faithfulness_text_img_scores = []

    for key in evaluations:
        correctness_scores.append(evaluations[key]["correctness_img_result"]["score"])
        relevancy_scores.append(evaluations[key]["relevancy_img_result"]["score"])
        faithfulness_scores.append(evaluations[key]["faithfulness_img_result"]["score"])
        correctness_text_img_scores.append(evaluations[key]["correctness_text_img_result"]["score"])
        relevancy_text_img_scores.append(evaluations[key]["relevancy_text_img_result"]["score"])
        faithfulness_text_img_scores.append(evaluations[key]["faithfulness_text_img_result"]["score"])

    correctness_mean = sum(correctness_scores) / len(correctness_scores)
    relevancy_mean = sum(relevancy_scores) / len(relevancy_scores)
    faithfulness_mean = sum(faithfulness_scores) / len(faithfulness_scores)
    correctness_text_img_mean = sum(correctness_text_img_scores) / len(correctness_text_img_scores)
    relevancy_text_img_mean = sum(relevancy_text_img_scores) / len(relevancy_text_img_scores)
    faithfulness_text_img_mean = sum(faithfulness_text_img_scores) / len(faithfulness_text_img_scores)

    # to round to 3 decimal places
    correctness_mean = round(correctness_mean, 3)
    relevancy_mean = round(relevancy_mean, 3)
    faithfulness_mean = round(faithfulness_mean, 3)
    correctness_text_img_mean = round(correctness_text_img_mean, 3)
    relevancy_text_img_mean = round(relevancy_text_img_mean, 3)
    faithfulness_text_img_mean = round(faithfulness_text_img_mean, 3)

    print(f"Correctness mean: {correctness_mean}")
    print(f"Relevancy mean: {relevancy_mean}")
    print(f"Faithfulness mean: {faithfulness_mean}")
    print(f"Correctness text_img mean: {correctness_text_img_mean}")
    print(f"Relevancy text_img mean: {relevancy_text_img_mean}")
    print(f"Faithfulness text_img mean: {faithfulness_text_img_mean}")


