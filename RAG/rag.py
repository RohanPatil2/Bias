# rag_sideA.py

import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langgraph.graph import END, StateGraph
from langchain_ollama import OllamaLLM
from RAG.agent import AgentState  # TypedDict for the graph state


class RAG_chatbot:
    """
    A simplified RAG chatbot that retrieves relevant Antarctic ice documents
    and provides a neutral, evidence-based answer without taking a pre-defined side.
    """

    def __init__(self):
        # 1) Initialize Ollama LLM
        model_name = os.getenv("OLLAMA_MODEL", "hf.co/Rohanpatil02/FineTunedBias:latest")
        self.llm = OllamaLLM(model=model_name)

        # 2) Set up embeddings and Chroma retriever for PDFs
        embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        pdf_dir    = os.getenv("PDF_DIR", "RAG/data")
        chroma_dir = os.getenv("CHROMA_DIR", "RAG/chroma_data")
        self.retriever = self._build_retriever(pdf_dir, chroma_dir)

        # 3) Chat history storage
        self.store = {}

        # 4) Build the QA chain and the state graph
        self._init_qa_chain()
        self._build_graph()

    def _build_retriever(self, data_dir: str, persist_dir: str):
        # Load and chunk all PDFs in the data directory
        docs = []
        if os.path.isdir(data_dir):
            for fn in os.listdir(data_dir):
                if not fn.lower().endswith(".pdf"):
                    continue
                path = os.path.join(data_dir, fn)
                try:
                    docs.extend(PyPDFLoader(path).load())
                except Exception:
                    try:
                        docs.extend(UnstructuredPDFLoader(path).load())
                    except Exception:
                        logging.warning(f"Skipping corrupt or unreadable PDF: {fn}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Persist into Chroma
        Chroma.from_documents(chunks, embedding=self.embeddings, persist_directory=persist_dir)
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        ).as_retriever()

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        # Retrieve or create chat history for the session
        self.store.setdefault(session_id, ChatMessageHistory())
        return self.store[session_id]

    def _init_qa_chain(self):
        # Neutral system prompt instructing balanced, evidence-based summaries
        sys_prompt = (
            "You are an expert assistant. "
            "Given the following retrieved documents on Antarctic ice mass changes, "
            "provide a factual, balanced summary of what the data supports. "
            "Do not assume any position beyond the evidence—if the documents conflict, present both perspectives. "
            "If the evidence is insufficient, state that clearly.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        stuff_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, stuff_chain)

        self.chatbot = RunnableWithMessageHistory(
            self.qa_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _retrieve_docs(self, state: AgentState):
        # Retrieve the top-k relevant document chunks
        docs = self.retriever.get_relevant_documents(state["question"])
        state["documents"] = [d.page_content for d in docs]
        return state

    def _generate_answer(self, state: AgentState):
        # Generate the answer using the QA chain
        reply = self.chatbot.invoke(
            {"input": state["question"]},
            config={"configurable": {"session_id": "neutral"}}
        )
        state["llm_output"] = reply["answer"]
        return state

    def _build_graph(self):
        # Build a simple retrieval→generation graph
        g = StateGraph(AgentState)
        g.add_node("retrieve_docs",    self._retrieve_docs)
        g.add_node("generate_answer",  self._generate_answer)
        g.add_edge("retrieve_docs", "generate_answer")
        g.add_edge("generate_answer", END)
        g.set_entry_point("retrieve_docs")
        self.app = g.compile()

    def ask(self, question: str) -> str:
        # Synchronous API: ask your question and get back a balanced answer
        state = self.app.invoke({"question": question})
        return state["llm_output"]


if __name__ == "__main__":
    bot = RAG_chatbot()
    question = "What does the research say about Antarctic ice mass changes in recent decades?"
    print("Q:", question)
    print("A:", bot.ask(question))
