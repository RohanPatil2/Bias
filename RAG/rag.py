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
    A simplified RAG chatbot that always retrieves from Side A papers
    (Antarctic ice melting as part of recurring natural freeze–melt cycles)
    and answers in that frame.
    """

    def __init__(self):
        # 1) Initialize Ollama LLM
        model_name = os.getenv("OLLAMA_MODEL", "hf.co/Rohanpatil02/ChatB:latest")
        self.llm = OllamaLLM(model=model_name)

        # 2) Set up embeddings and Chroma retriever for Side A PDFs
        embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        pdf_dir    = os.getenv("SIDEA_PDF_DIR", "RAG/sideb")
        chroma_dir = os.getenv("SIDEA_CHROMA_DIR", "RAG/sideb")
        self.sideA_retriever = self._build_sideA_retriever(pdf_dir, chroma_dir)

        # 3) Chat history storage
        self.store = {}

        # 4) Build the QA chain and the state graph
        self._init_qa_chain()
        self._build_graph()

    def _build_sideA_retriever(self, data_dir: str, persist_dir: str):
        # Load and chunk all Side A PDFs
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
                        logging.warning(f"Skipping corrupt PDF: {fn}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Persist into Chroma
        Chroma.from_documents(chunks, embedding=self.embeddings, persist_directory=persist_dir)
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        ).as_retriever()

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        # Retrieve or create chat history
        self.store.setdefault(session_id, ChatMessageHistory())
        return self.store[session_id]

    def _init_qa_chain(self):
        # System prompt focused exclusively on the natural-cycle argument
        sys_prompt = (
            "You are a climate-science expert arguing that Antarctic sea ice melt "
            "is a recurring natural phenomenon driven by seasonal freeze–thaw cycles. "
            "Use the following retrieved Side A documents to support that claim. "
            "If you lack evidence, say you don't know.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        stuff_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.qa_chain = create_retrieval_chain(self.sideA_retriever, stuff_chain)

        self.chatbot = RunnableWithMessageHistory(
            self.qa_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _retrieve_docs(self, state: AgentState):
        # Retrieve the top-k Side A chunks
        docs = self.sideA_retriever.get_relevant_documents(state["question"])
        state["documents"] = [d.page_content for d in docs]
        return state

    def _generate_answer(self, state: AgentState):
        # Generate the answer using the QA chain
        reply = self.chatbot.invoke(
            {"input": state["question"]},
            config={"configurable": {"session_id": "sideA"}}
        )
        state["llm_output"] = reply["answer"]
        return state

    def _build_graph(self):
        # Build a simple 2-node graph: retrieve -> generate -> END
        g = StateGraph(AgentState)
        g.add_node("retrieve_docs",    self._retrieve_docs)
        g.add_node("generate_answer",  self._generate_answer)
        g.add_edge("retrieve_docs", "generate_answer")
        g.add_edge("generate_answer", END)
        g.set_entry_point("retrieve_docs")
        self.app = g.compile()

    def ask(self, question: str) -> str:
        # Synchronous API to ask your question
        state = self.app.invoke({"question": question})
        return state["llm_output"]


if __name__ == "__main__":
    bot = RAG_chatbot()
    question = "Why is the Antarctic sea ice undergoing seasonal melt and refreeze?"
    print("Q:", question)
    print("A:", bot.ask(question))
