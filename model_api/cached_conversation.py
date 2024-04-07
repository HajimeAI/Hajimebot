import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

from configs import (
    logger, PROMPT_TEMPLATES, OPENAI_PROXY,
    LOCAL_LLM_SERVER, LOCAL_LLM_MODEL, ChatModels, DEFAULT_CHAT_MODEL
)

class UnsupportedChatModelException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f'UnsupportedChatModelException: {self.message}'

class CachedConversation():
    def __init__(self, model: ChatModels=None):
        temperature = 0
        if model is None:
            model = DEFAULT_CHAT_MODEL
        self._model = model
        if model == ChatModels.OPENAI:
            self._llm = ChatOpenAI(
                temperature=temperature,
                api_key=os.environ['CHAT_OPENAI_KEY'],
                base_url='https://api.aiproxy.io/v1',
                # base_url='https://p2p.hajime.ai/openai-api/v1',
                model='gpt-4',
                # model='gpt-3.5-turbo',
                openai_proxy=OPENAI_PROXY,
            )
        elif model == ChatModels.OLLAMA:
            self._llm = ChatOllama(
                temperature=temperature,
                base_url=LOCAL_LLM_SERVER, 
                model=LOCAL_LLM_MODEL,
            )
        else:
            raise UnsupportedChatModelException(str(model))
        
        self._memory_simple_chat = ConversationBufferWindowMemory(
            return_messages=True, 
            input_key='query',
            k=3,
        )
        # self._memory_kb_chat = ConversationBufferWindowMemory(
        #     return_messages=True, 
        #     input_key='query',
        #     k=3,
        # )
        self._verbose = True

    def simple_chat(self, query: str, memory=None):
        # prompt = PromptTemplate.from_template(
        #     PROMPT_TEMPLATES['chat']
        # )
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a nice chatbot having a conversation with a human."
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{query}")
            ]
        )

        if not memory:
            memory = self._memory_simple_chat

        conversation = LLMChain(
            llm=self._llm, 
            prompt=prompt, 
            memory=memory, 
            verbose=self._verbose,
        )
        return conversation.predict(query=query)

    def kb_chat(self, query: str, reference: str):
        prompt = PromptTemplate.from_template(
            PROMPT_TEMPLATES['kb_chat']
        )
        conversation = LLMChain(
            llm=self._llm, 
            prompt=prompt, 
            verbose=self._verbose
        )
        answer = conversation.predict(query=query, reference=reference)
        if answer != 'NOT_FOUND':
            return answer
        
        return self.simple_chat(query)
