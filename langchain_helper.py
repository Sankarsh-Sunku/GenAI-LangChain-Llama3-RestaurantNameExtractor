from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain # Import SequentialChain

prompt_template_name = PromptTemplate(
    input_variables =['cuisine'],
    template="   I want to open a restaurant for {cuisine} food. Suggest a fency name for this."
)

prompt_template_items = PromptTemplate(
    input_variables = ['restaurant_name'],
    template="List some menu items for a restaurant named {restaurant_name}."
)


def generate_restaurant_name_and_items(cuisine):
    sec_key = "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"
    login(token=sec_key)
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # Name of the model which you are working
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.7, model_kwargs={"max_length": 512, "token": sec_key})
    # as the temperature is increased it will take risks and be more creative to provide the results for our prompt

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', "menu_items"]
    )

    response = chain({"cuisine": cuisine})

    return {
        'restaurantName' : response['restaurant_name'],
        'menuItems' : response['menu_items']
    }
