import os
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.app.services.pubmed_service import pubmed_service
from backend.app.services.graph_service import graph_service
from backend.app.services.llm_service import llm_service
from backend.app.services.text_processing import normalize_medical_text, validate_entity
from backend.app.models.schemas import ExtractionResponse, SearchResponse, SearchResult
from backend.app.core.config import settings

class RAGService:
    def __init__(self):
        self.llm_service = llm_service
        
        # Initialize Embeddings and Vector Store
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        
        self.vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
            collection_name="vimed_rag"
        )
        print("Vector Store initialized.")
    
    async def ingest_document(self, file_path: str):
        """Ingest a document: Load -> Chunk -> Extract -> Update Graph & Vector Store"""
        print(f"Ingesting: {file_path}")
        
        # 1. Load and Chunk
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        # Filter first few pages if needed (as per notebook)
        # documents = documents[16:] 
        
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print(f"Total chunks: {len(chunks)}")
        
        # 2. Add to Vector Store
        print("Adding chunks to Vector Store...")
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        
        # 3. Extract and Update Graph
        parser = PydanticOutputParser(pydantic_object=ExtractionResponse)
        
        def extraction_chain_factory(llm):
            return PromptTemplate(
                template="""Bạn là chuyên gia trích xuất Knowledge Graph y tế.
                
                Trích xuất các thực thể (Entity) và quan hệ (Relation) từ văn bản sau.
                
                RULES:
                - Entity types: DISEASE, DRUG, SYMPTOM, TEST, ANATOMY, TREATMENT, PROCEDURE, RISK_FACTOR, LAB_VALUE
                - Relation types: CAUSES, TREATS, PREVENTS, DIAGNOSES, SYMPTOM_OF, COMPLICATION_OF, SIDE_EFFECT_OF, INCREASES_RISK, INTERACTS_WITH, WORSENS, INDICATES, RELATED_TO
                - Confidence: 1-10
                
                TEXT: {text}
                
                {format_instructions}
                """,
                input_variables=["text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            ) | llm | parser

        for i, chunk in enumerate(chunks):
            # Skip if already processed (check graph service)
            # For now, we process all. In production, we should check last_chunk_id.
            
            try:
                result = self.llm_service.execute_chain(extraction_chain_factory, {"text": chunk.page_content})
                
                if result:
                    page_num = chunk.metadata.get('page', 0)
                    
                    # Add Entities
                    for entity in result.entities:
                        # Validation logic here if needed
                        graph_service.add_entity(entity, page_num, i)
                        
                    # Add Relations
                    for relation in result.relations:
                        # Validation logic here if needed
                        graph_service.add_relation(relation, page_num, i)
                        
                    print(f"Chunk {i+1}/{len(chunks)}: +{len(result.entities)} entities, +{len(result.relations)} relations")
                    
                    # Checkpoint periodically
                    if (i + 1) % 20 == 0:
                        graph_service.save_checkpoint(i, len(chunks))
            
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
        
        # Final Save
        graph_service.save_checkpoint(len(chunks), len(chunks))
        return {"status": "success", "chunks_processed": len(chunks)}

    async def process_question(self, question: str) -> Dict[str, Any]:
        """Answer question using RAG (Vector + Graph + PubMed)"""
        print(f"Processing question: {question}")
        
        # 1. Search PubMed
        pubmed_docs = pubmed_service.search(question, max_results=2)
        pubmed_context = "\n\n".join(pubmed_docs) if pubmed_docs else "No external context found."
        
        # 2. Vector Search (Semantic Search)
        print("Searching Vector Store...")
        vector_docs = self.vectorstore.similarity_search(question, k=3)
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs]) if vector_docs else "No vector context found."
        
        # 3. Search Local Graph with Reasoning
        from backend.app.services.reasoning_service import reasoning_service
        
        graph_context = ""
        norm_question = normalize_medical_text(question)
        
        # Identify extraction targets (simple keyword match for now, could use LLM to extract entities first)
        found_entities = []
        # Try to find strict matches first (multi-word)
        for node in graph_service.graph.nodes():
            if normalize_medical_text(node) in norm_question:
                found_entities.append(node)
                
        # If no strict matches, try individual words
        if not found_entities:
            for word in norm_question.split(): 
                word_title = word.title()
                if graph_service.graph.has_node(word_title):
                    found_entities.append(word_title)
        
        # Deduplicate
        found_entities = list(set(found_entities))
        
        if found_entities:
            print(f"Found graph entities: {found_entities}")
            for entity in found_entities:
                # Use deep reasoning for found entities
                reasoning = reasoning_service.reason_about_entity(entity, context_depth=2)
                graph_context += f"{reasoning}\n"
        else:
            graph_context = "No directly related entities found in Graph."
        
        full_context = f"""
        Internal Document Knowledge (Vector Search):
        {vector_context}
        
        Internal Graph Knowledge (Reasoning):
        {graph_context}
        
        External PubMed Knowledge:
        {pubmed_context}
        """
        
        # 4. Generate Answer
        answer_prompt = PromptTemplate(
            template="""Answer the following medical question using the provided context.
            Identify conflicting information if any. Prioritize Internal Document Knowledge.
            
            Question: {question}
            
            Context:
            {context}
            
            Answer (in Vietnamese):""",
            input_variables=["question", "context"]
        )
        
        try:
            res = self.llm_service.execute_chain(
                lambda llm: answer_prompt | llm,
                {"question": question, "context": full_context}
            )
            answer = res.content if hasattr(res, 'content') else str(res)
        except Exception as e:
            answer = f"Error generating answer: {e}"

        return {
            "answer": answer,
            "context": full_context,
            "graph_visual_url": "/api/v1/graph/visualize"
        }


rag_service = RAGService()
