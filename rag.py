# import required dependencies
# https://docs.chainlit.io/integrations/langchain
import os
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain_core.messages import HumanMessage, SystemMessage

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "chroma")


# Set up RetrievelQA model
rag_prompt_mistral = hub.pull("rlm/rag-prompt-mistral")


def load_model():
    llm = Ollama(
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def get_retriever():
    DB_PATH = DB_DIR
    vectorstore = Chroma(
        persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model="mistral")
    )

    # Create retriever
    retriever = vectorstore.as_retriever(k=3)

    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_elements(docs):

    return [
        cl.Text(
            name="From: " + doc.metadata.get("id", None),
            content=doc.page_content,
            display="inline",
        )
        for doc in docs
    ]


# Prompt
rag_prompt = """You are an assistant for question-answering tasks. 

    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context. 

    Use three sentences maximum and keep the answer concise.

    Answer:
    
    
    """


@cl.on_chat_start
async def start():
    retriever = get_retriever()
    llm = load_model()

    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = rag_prompt
    await welcome_message.update()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("llm", llm)


@cl.on_message
async def main(message):

    text_elements = []  # type: List[cl.Text]
    msg = cl.Message(content="", elements=text_elements)
    await msg.send()

    retriever = cl.user_session.get("retriever")
    llm = cl.user_session.get("llm")

    question = message.content
    docs = retriever.invoke(question)
    docs_txt = format_docs(docs)

    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    answer = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    print("answer:", answer, "type:", type(answer))

    await msg.stream_token(answer)

    text_elements = format_elements(docs)
    await cl.Message(
        content="Check out this text element!",
        elements=text_elements,
    ).send()

    await msg.update()
