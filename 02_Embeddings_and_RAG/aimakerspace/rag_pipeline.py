from typing import Dict, List, Any, Optional
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt


class EnhancedRetrievalAugmentedQAPipeline:
    """
    Enhanced RAG Pipeline with metadata support for better source attribution and filtering.
    """
    
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False,
                 include_metadata: bool = True) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.response_style = response_style
        self.include_scores = include_scores
        self.include_metadata = include_metadata
        
        # Enhanced RAG system template with metadata awareness
        self.rag_system_template = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific sources when possible using [Source N] format
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- When citing sources, include relevant file information when available
- Only provide answers when you are confident the context supports your response."""

        # Enhanced user template with metadata information
        self.rag_user_template = """Context Information:
{context}

Source Information:
{source_info}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above, citing sources using [Source N] format."""

        self.rag_system_prompt = SystemRolePrompt(self.rag_system_template)
        self.rag_user_prompt = UserRolePrompt(self.rag_user_template)

    def run_pipeline(self, user_query: str, k: int = 4, 
                    filter_criteria: Optional[Dict[str, Any]] = None,
                    **system_kwargs) -> Dict[str, Any]:
        """
        Run the enhanced RAG pipeline with metadata support.
        
        Args:
            user_query: The question to answer
            k: Number of context chunks to retrieve
            filter_criteria: Optional metadata filtering criteria
            **system_kwargs: Additional system prompt parameters
            
        Returns:
            Dictionary containing response, context, metadata, and source information
        """
        # Retrieve relevant contexts with metadata
        context_results = self.vector_db_retriever.search_by_text(
            user_query, 
            k=k, 
            filter_criteria=filter_criteria,
            return_metadata=self.include_metadata
        )
        
        # Format context and source information
        context_prompt = ""
        source_info = ""
        similarity_scores = []
        
        for i, result in enumerate(context_results, 1):
            if self.include_metadata and len(result) == 3:
                context, score, metadata = result
                context_prompt += f"[Source {i}]: {context}\n\n"
                
                # Format source information
                source_details = []
                if metadata.get('source_file'):
                    source_details.append(f"File: {metadata['source_file']}")
                if metadata.get('file_type'):
                    source_details.append(f"Type: {metadata['file_type']}")
                if metadata.get('chunk_index') is not None:
                    source_details.append(f"Chunk: {metadata['chunk_index']}")
                if metadata.get('page_range'):
                    source_details.append(f"Pages: {metadata['page_range']}")
                
                source_info += f"Source {i}: {', '.join(source_details)}\n"
                
            else:
                # Fallback for when metadata is not available
                context, score = result[:2]
                context_prompt += f"[Source {i}]: {context}\n\n"
                source_info += f"Source {i}: No metadata available\n"
            
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": system_kwargs.get("response_length", "detailed")
        }
        
        formatted_system_prompt = self.rag_system_prompt.create_message(**system_params)
        
        # Create user message with context and source information
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "source_info": source_info.strip(),
            "context_count": len(context_results),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = self.rag_user_prompt.create_message(**user_params)

        # Generate response
        response = self.llm.run([formatted_system_prompt, formatted_user_prompt])
        
        return {
            "response": response,
            "context": context_results,
            "context_count": len(context_results),
            "source_info": source_info,
            "similarity_scores": similarity_scores if self.include_scores else None,
            "filter_criteria": filter_criteria,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }
    
    def search_with_filters(self, query: str, k: int = 4, 
                          file_type: Optional[str] = None,
                          source_file: Optional[str] = None) -> List:
        """
        Convenience method for common filtering scenarios.
        
        Args:
            query: Search query
            k: Number of results
            file_type: Filter by file type (txt, pdf)
            source_file: Filter by specific source file
            
        Returns:
            Search results with metadata
        """
        filter_criteria = {}
        if file_type:
            filter_criteria["file_type"] = file_type
        if source_file:
            filter_criteria["source_file"] = source_file
            
        return self.vector_db_retriever.search_by_text(
            query, k=k, 
            filter_criteria=filter_criteria if filter_criteria else None,
            return_metadata=True
        )