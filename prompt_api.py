from langchain_community.chat_models import ChatOllama
from transformers import file_utils
print(file_utils.default_cache_path)

# Initialize the ChatOllama instance with the model name
langchain_llm = ChatOllama(model="llama3:latest")

# Define a function to interact with the model using the correct message format
def interact_with_llama3(prompt):
    # Create a message in the expected format
    messages = [{"role": "user", "content": prompt}]
    response = langchain_llm.invoke(messages)
    return response.content

# Example interaction
prompt = "あなたは日本語しゃべられますよね？"
response = interact_with_llama3(prompt)
print("AI response:", response)
