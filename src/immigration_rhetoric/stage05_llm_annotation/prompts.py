import json


def prompt_step_1(snippet: str) -> str:
    """Ask the model to explain how a reader might feel towards immigration/immigrants after reading `snippet`."""
    return f"""

    **Task Overview**

    You are reading a snippet from a speech by a British MP. The snippet might concern **immigration or immigrants**. If the snippet is indeed **about immigration or immigrants**, explain concisely how someone might feel towards **immigration or immigrants** after reading it and why. Does it use fearmongering to promote tough measures to control immigration? Or does it express a compassionate attitude towards immigration and/or immigrants? Note that it will not always be explicitly or directly mentioned in the snippet.

    ```{snippet}```
    """


def prompt_step_2(step_1_response: str) -> str:
    """Ask the model to assign a probability (very low/low/moderate/high/very high) to each of labels A-D."""
    return f"""

    **Task Overview**

    You are reading the explanation of how someone might feel towards immigration or immigrants after reading a snippet from a political speech by a British MP.  Based on this reasoning, assign a probability (very low/low/moderate/high/very high) for each label: A, B, C, D.

    Where:

    **A**: Assign this label when the text expresses support, advocacy, or overall positive sentiment towards immigration and immigrants. This may be explicit or implicit. For example, when politicians talk about the positive impact of immigration on society, economy, cultural diversity and promote integration. Also, consider instances where the text counters fear-based narratives about immigration or immigrants.

    **B**: Assign this label when the text contains language that portrays immigration or immigrants as a threat, or expresses concerns about security risks, strains on public resources, or cultural tensions attributed to immigration or immigrants. Also, consider instances where the text advocates for assimilation, as opposed to integration, restrictive immigration policies, or measures that are detrimental to immigrants' rights or well-being. This may be explicit or implicit.

    **C**: Use this label when the text presents factual information or discusses immigration-related issues without expressing a clear positive or negative sentiment. This may include discussions about immigration policies, demographic changes, or societal challenges associated with immigration, presented in a balanced and impartial manner without endorsing or criticising specific viewpoints.

    **D**: Apply this label when the snippet of text is not related to immigration or does not contain meaningful information that contributes to understanding the discourse around immigration. This label should be used for text that is off-topic, contains irrelevant content, or is too ambiguous to be categorised accurately with the other labels.

    Provide the response in JSON format ONLY with the following keys:
    "label": "probability"

    ```{step_1_response}```
    """


def clean_step2_response(response: str) -> dict:
    """Strip escape characters/surrounding text from a step-2 response and parse the remaining JSON object."""
    cleaned_string = response.replace("\n", "").replace("json", "").replace("```", "")
    json_response = "{" + cleaned_string.split("{")[-1].split("}")[0] + "}"
    return json.loads(json_response)


def highest_score_labels(dictionary: dict) -> list:
    """Return the label(s) with the highest assigned probability, in the order very low -> very high."""
    scoring_order = ["very low", "low", "moderate", "high", "very high"]

    max_score = float("-inf")
    result = []
    for key, value in dictionary.items():
        score = scoring_order.index(value)
        if score > max_score:
            max_score = score
            result = [key]
        elif score == max_score:
            result.append(key)
    return result


def convert_alphabetic_labels(alphabetic_labels: list) -> list:
    """Map A/B/C/D labels to their human-readable names."""
    replacement_labels = {
        "A": "POSITIVE_TOWARDS_IMMIGRATION",
        "B": "NEGATIVE_TOWARDS_IMMIGRATION",
        "C": "NEUTRAL_TOWARDS_IMMIGRATION",
        "D": "NOT_ABOUT_IMMIGRATION",
    }
    return [replacement_labels[label] for label in alphabetic_labels]
