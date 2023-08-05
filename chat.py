import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import load

load_dotenv()

# key = os.getenv("QDRANT_API_KEY")
# host = os.getenv("QDRANT_HOST")  # This is a cloud host, try local too
# client = QdrantClient(url=host, api_key=key)
doc_store = load.load_text()
retriever = doc_store.as_retriever()
chat = ChatOpenAI(temperature=0.2)
template = """
    You are a helpful assistant that helps Dungeons and dragons players
    recall their previous adventures and game activity.
    Use the following input to help you answer questions:

    {quest_context}

    Provide as much detail about the actions immediately
    before and after the event in question}
    """
human_template = "{text}"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)
# The llm chat question is not a good way to get the context from the db
query = "What happened when we arrived in skullport?"
context = retriever.retrieve(query)
chat_prompt = prompt_template.format_prompt(
    quest_context=context,
)
chain = RetrievalQA.from_llm(
    llm=chat,
    prompt=chat_prompt,
    input_keys=["quest_context", "text"],
)
response = chain.run(context, query)

print(response)
