from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from backend.app.services.pubmed_service import pubmed_service
from backend.app.services.graph_service import graph_service
from backend.app.services.llm_service import llm_service
from backend.app.core.config import settings

class RAGService:
    def __init__(self):
        self.llm_service = llm_service

    async def process_question(self, question: str) -> Dict[str, Any]:
        """Main pipeline to answer a medical question"""
        print(f"Processing question: {question}")
        
        # 1. Search PubMed
        search_results = pubmed_service.search(question, max_results=2)
        context = "\n".join(search_results) if search_results else "No external context found."
        
        # 2. Extract Entities
        entities = []
        try:
            entity_schemas = [
                ResponseSchema(name="entities", description="List of medical entities", type="array"),
                ResponseSchema(name="scores", description="Relevance scores (1-10)", type="array"),
                ResponseSchema(name="descriptions", description="Brief descriptions", type="array")
            ]
            entity_parser = StructuredOutputParser.from_response_schemas(entity_schemas)
            
            def entity_chain_factory(llm):
                return PromptTemplate(
                   template="""Extract medical entities from this question and context.
                   Question: {question}
                   Context: {context}
                   {format_instructions}""",
                   input_variables=["question", "context"],
                   partial_variables={"format_instructions": entity_parser.get_format_instructions()}
                ) | llm | entity_parser

            entities_res = self.llm_service.execute_chain(
                entity_chain_factory, 
                {"question": question, "context": context}
            )
            entities = entities_res.get("entities", [])
            
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            entities = ["Error extraction"]

        # 3. Answer Question
        answer = "Could not generate answer."
        try:
            def answer_chain_factory(llm):
                return PromptTemplate(
                    template="""Answer the question based on the context.
                    Question: {question}
                    Context: {context}
                    Answer included medical reasoning:""",
                    input_variables=["question", "context"]
                ) | llm 
            
            res = self.llm_service.execute_chain(
                answer_chain_factory, 
                {"question": question, "context": context}
            )
            
            if hasattr(res, 'content'):
                answer = res.content
            else:
                answer = str(res)
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = f"Error: {str(e)}"

        return {
            "answer": answer,
            "context": context,
            "entities": entities,
            "graph_html": "/data/current_graph.html"
        }

rag_service = RAGService()
