from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator

from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI 

from langchain.memory import PostgresChatMessageHistory

from langchain.chains import OpenAIModerationChain, SequentialChain, LLMChain, SimpleSequentialChain


class CustomModeration(OpenAIModerationChain):
    """
    Moderation chains are useful for 
    detecting text that could be hateful, 
    violent, etc. This can be useful to 
    apply on both user input, but also on 
    the output of a Language Model.
    
    Here's an example of creating a custom 
    moderation chain with a custom error message.
    """
    
    def _moderate(self, text: str, results: dict) -> str:
        if results["flagged"]:
            error_str = f"The following text was found that violates OpenAI's content policy: {text}"
            return error_str
        return text
    
# custom_moderation = CustomModeration()
moderation_chain = OpenAIModerationChain(error=True)


class DocBasicQA:
    def __init__(self, file_path):
        self.postgres_uri = "postgresql://postgres:mypassword@localhost/chat_history"
        self.postgres_session_id = "foo"
        self.texts = self.get_text(file_path)
        
        # Moderation check
        self.moderation_chain = OpenAIModerationChain(error=True)
    
    def get_text(self, file_path):
        with open(file_path) as f:
            state_of_the_union = f.read()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(state_of_the_union)
        return texts 
      
    def get_relevant_doc(self, text, query):
        # For QA with sources 
        # return docsearch.similarity_search(query)
        
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
        return docsearch.get_relevant_documents(query)
    
    def get_output(self, rel_doc, query):
        # For QA with sources 
        # chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce")
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce")
        output = chain({"input_documents": rel_doc, "question": query}, return_only_outputs=True)
        return output["output_text"]
    
    def get_output__custom_temp(self, **kwargs):
        """
        This generate a reply from a custom prompt templates (question_prompt, combine_prompt)
        
        parameters
            qn_prompt_template -> text 
            qn_prompt_variables -> variables for qn_prompt_template
            
            cb_prompt_template -> text 
            cb_prompt_variables -> variables for cb_prompt_template
            
            rel_doc -> get_relevant_doc instance
            query -> text query
        """
        QUESTION_PROMPT = PromptTemplate(
            template=kwargs["qn_prompt_template"], 
            input_variables=kwargs["qn_prompt_variables"]
        )
        
        COMBINE_PROMPT = PromptTemplate(
            template=kwargs["cb_prompt_template"],
            input_variables=kwargs["cb_prompt_variables"]
        )
        # For QA with sources 
        # chain = load_qa_with_sources_chain(OpenAI(temperature=0), chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce", return_map_steps=True, question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
        output = chain({
            "input_documents": kwargs["rel_doc"], 
            "question": kwargs["query"]
        }, return_only_outputs=True)
        return output["output_text"]
    
    def save_message(self, user_msg, ai_msg):
        history = PostgresChatMessageHistory(
            connection_string=self.postgres_uri,
            session_id=self.postgres_session_id,
        )
        
        history.add_user_message(user_msg)
        history.add_ai_message(ai_msg)
        return history.messages
    
    def run(query, **kwargs):
        """
        keywords parameters
            qn_prompt_template -> text 
            qn_prompt_variables -> variables for qn_prompt_template
            
            cb_prompt_template -> text 
            cb_prompt_variables -> variables for cb_prompt_template
        """
        
        try:
            query = self.moderation_chain.run(query)
        except ValueError:
            return f"Text was found that violates OpenAI's content policy"
        
        rel_doc = self.get_relevant_doc(self.texts, query)
        
        if kwargs:
            kwargs["rel_doc"] = rel_doc
            kwargs["query"] = query
            output = self.get_output__custom_temp(kwargs)
        else:
            output = self.get_output(rel_doc, query)
        
        self.save_message(query, output)
        return output


class DocRetrieverQA:
    def __init__(self, file_path):
        self.postgres_uri = "postgresql://postgres:mypassword@localhost/chat_history"
        self.postgres_session_id = "foo"
        self.text = self.get_text(file_path)
        # Moderation check
        self.moderation_chain = OpenAIModerationChain(error=True)
    
    def get_text(self, file_path):
        with open(file_path) as f:
            state_of_the_union = f.read()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(state_of_the_union)
        return text 
    
    def save_message(self, user_msg, ai_msg):
        history = PostgresChatMessageHistory(
            connection_string=self.postgres_uri,
            session_id=self.postgres_session_id,
        )
        
        history.add_user_message(user_msg)
        history.add_ai_message(ai_msg)
        return history.messages
        
    def run(query):
        try:
            query = self.moderation_chain.run(query)
        except ValueError:
            return f"Text was found that violates OpenAI's content policy"
        
        for i, text in enumerate(self.texts):
            text.metadata['source'] = f"{i}-pl"
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)
        
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
        
        qa_chain = create_qa_with_sources_chain(llm)
        
        doc_prompt = PromptTemplate(
            template="Content: {page_content}\nSource: {source}",
            input_variables=["page_content", "source"],
        )
        
        final_qa_chain = StuffDocumentsChain(
            llm_chain=qa_chain, 
            document_variable_name='context',
            document_prompt=doc_prompt,
        )
        
        retrieval_qa = RetrievalQA(
            retriever=docsearch.as_retriever(),
            combine_documents_chain=final_qa_chain
        )
        output = retrieval_qa.run(query)
        self.save_message(query, output)
        return output