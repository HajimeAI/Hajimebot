from typing import List, Optional
import sys
import asyncio
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain, MapReduceDocumentsChain

from configs import (logger)
from knowledge_base.model.kb_document_model import DocumentWithVSId


class SummaryAdapter:
    _OVERLAP_SIZE: int
    token_max: int
    _separator: str = "\n\n"
    chain: MapReduceDocumentsChain

    def __init__(self, overlap_size: int, token_max: int,
                 chain: MapReduceDocumentsChain):
        self._OVERLAP_SIZE = overlap_size
        self.chain = chain
        self.token_max = token_max

    @classmethod
    def form_summary(cls,
                     llm: BaseLanguageModel,
                     reduce_llm: BaseLanguageModel,
                     overlap_size: int,
                     token_max: int = 1300):
        # This controls how each document will be formatted. Specifically,
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        # The prompt here should take as an input variable the
        # `document_variable_name`
        prompt_template = (
            "Task brief:" 
            "{task_briefing}" 
            "Document content: "
            "\r\n"
            "{context}"
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["task_briefing", "context"]
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        # We now define how to combine these summaries
        reduce_prompt = PromptTemplate.from_template(
            "Combine these summaries: {context}"
        )
        reduce_llm_chain = LLMChain(llm=reduce_llm, prompt=reduce_prompt)

        document_variable_name = "context"
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name
        )
        reduce_documents_chain = ReduceDocumentsChain(
            token_max=token_max,
            combine_documents_chain=combine_documents_chain,
        )
        chain = MapReduceDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name=document_variable_name,
            reduce_documents_chain=reduce_documents_chain,
            return_intermediate_steps=True
        )
        return cls(overlap_size=overlap_size,
                   chain=chain,
                   token_max=token_max)

    def summarize(self,
                  file_description: str,
                  docs: List[DocumentWithVSId] = []
                  ) -> List[Document]:

        if sys.version_info < (3, 10):
            loop = asyncio.get_event_loop()
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.asummarize(file_description=file_description,
                                                       docs=docs))

    async def asummarize(self,
                         file_description: str,
                         docs: List[DocumentWithVSId] = []) -> List[Document]:

        logger.info("start summary")
        summary_combine, summary_intermediate_steps = self.chain.combine_docs(docs=docs,
                                                                              task_briefing="task brief")
        print(summary_combine)
        print(summary_intermediate_steps)

        # if len(summary_combine) == 0:
        #     result_docs = [
        #         Document(page_content=question_result_key, metadata=docs[i].metadata)
        #         # This uses metadata from the docs, and the textual results from `results`
        #         for i, question_result_key in enumerate(
        #             summary_intermediate_steps["intermediate_steps"][
        #             :len(summary_intermediate_steps["intermediate_steps"]) // 2
        #             ])
        #     ]
        #     summary_combine, summary_intermediate_steps = self.chain.reduce_documents_chain.combine_docs(
        #         result_docs, token_max=self.token_max
        #     )
        logger.info("end summary")
        doc_ids = ",".join([doc.id for doc in docs])
        _metadata = {
            "file_description": file_description,
            "summary_intermediate_steps": summary_intermediate_steps,
            "doc_ids": doc_ids
        }
        summary_combine_doc = Document(page_content=summary_combine, metadata=_metadata)

        return [summary_combine_doc]

    def _drop_overlap(self, docs: List[DocumentWithVSId]) -> List[str]:
        merge_docs = []

        pre_doc = None
        for doc in docs:
            if len(merge_docs) == 0:
                pre_doc = doc.page_content
                merge_docs.append(doc.page_content)
                continue

            for i in range(len(pre_doc), self._OVERLAP_SIZE // 2 - 2 * len(self._separator), -1):
                pre_doc = pre_doc[1:]
                if doc.page_content[:len(pre_doc)] == pre_doc:
                    merge_docs.append(doc.page_content[len(pre_doc):])
                    break

            pre_doc = doc.page_content

        return merge_docs

    def _join_docs(self, docs: List[str]) -> Optional[str]:
        text = self._separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text


if __name__ == '__main__':

    docs = [
        'And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.',
    ]
    _OVERLAP_SIZE = 1
    separator: str = "\n\n"
    merge_docs = []
    pre_doc = None
    for doc in docs:
        if len(merge_docs) == 0:
            pre_doc = doc
            merge_docs.append(doc)
            continue

        for i in range(len(pre_doc), _OVERLAP_SIZE - 2 * len(separator), -1):
            pre_doc = pre_doc[1:]
            if doc[:len(pre_doc)] == pre_doc:
                page_content = doc[len(pre_doc):]
                merge_docs.append(page_content)

                pre_doc = doc
                break

    text = separator.join(merge_docs)
    text = text.strip()

    print(text)
