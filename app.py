from openai import OpenAI
from dotenv import load_dotenv
import os


def prompts_read(path):
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def response_generate(client, model, system_message, user_message, temperature):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content


def llm_connection(link, api_key, model, path):
    client = OpenAI(base_url=link, api_key=api_key)
    prompts = prompts_read(path)

    #user_question = input('Enter your question: ')

    user_question = 'Tell me the precepts of the student'
    system_message = "Always answer in rhymes."
    user_message = f"{user_question}\n\nContext: {prompts}"
    response = response_generate(client,
                                 model=model,
                                 system_message=system_message,
                                 user_message=user_message,
                                 temperature=0.7)

    print(response)


if __name__ == "__main__":
    load_dotenv()
    llm_url = os.getenv('LLM_URL')
    llm_model = os.getenv('LLM_MODEL')
    llm_checkpoint = os.getenv('LLM_CHECKPOINT')
    dataset_path = os.getenv('DATASET_PATH')
    llm_connection(llm_url, llm_model, llm_checkpoint, dataset_path)