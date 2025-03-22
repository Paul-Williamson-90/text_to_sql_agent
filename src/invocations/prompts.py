from llama_index.core import PromptTemplate

DEFAULT_STRUCTURED_PROMPT_TEMPLATE = PromptTemplate("{context}\n\n" "{schema}")

DEFAULT_CHOICE_VALIDATION_PROMPT_TEMPLATE = PromptTemplate(
    "# MASTER SYSTEM\n"
    "You are a master AI system that has AI students. "
    "Each student has been given the same instructions in the form of a prompt and has been task with generating an output in accordance with these instructions. "
    "As a master AI system, your task is to evaluate the outputs of your students and determine which output is the best. "
    "You will first be presented with the instructions that were given to the students, which you will find between these separators: "
    "-------------------STUDENT INSTRUCTIONS START-------------------\n"
    "-------------------STUDENT INSTRUCTIONS END-------------------\n"
    "You will then be presented with the outputs of the students, which you will find between these separators: "
    "-------------------STUDENT OUTPUT START-------------------\n"
    "-------------------STUDENT OUTPUT END-------------------\n"
    "You must evaluate the outputs of the students and determine which output is the best. "
    "You must provide a justification for your choice. "
    "You will choose the output by selecting the identifier of the output you believe is the best, this is in the form of a UUID located at the beginning of each student's output.\n\n"
    "We will now begin the evaluation process.\n\n"
    "-------------------STUDENT INSTRUCTIONS START-------------------\n"
    "{prompt}\n"
    "-------------------STUDENT INSTRUCTIONS END-------------------\n"
    "-------------------STUDENT OUTPUT START-------------------\n"
    "{choices}"
    "-------------------STUDENT OUTPUT END-------------------\n"
)
