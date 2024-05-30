from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from utils import database_managers, embedding, text_processing
import streamlit as st
from IPython.display import display, Markdown, Latex
import os
import subprocess
import platform

# Variables ###############################################################
load_dotenv(override=True)
COLLECTION_NAME = "pdf"
FOLDER_PATH = r"C:\\Users\\ELAFACRB1\\Codice\GitHub\\langchain-document-chatbot\\documents" if platform.system()=='Windows' else '/documents'
MODEL_NAME="Llama3-8b-8192"
TEMPERATURE=0

# Config ##################################################################
text_splitter=text_processing.TextSplitter()
embedding = embedding.EmbeddingFunction('fast-bgeEmbedding').embedder
vectore_store=qdrantClient = database_managers.QDrantDBManager(
    url=os.getenv('QDRANT_URL'),
    port=6333,
    collection_name=COLLECTION_NAME,
    vector_size=768,
    embedding=embedding,
    record_manager_url=r"sqlite:///record_manager_cache.sql"
)
vectore_store_client=vectore_store.vector_store
retriever = vectore_store_client.as_retriever()

# Functions ###############################################################
def research_papers(topic, n, year, pagine_scholar):
    with st.spinner("Ricerca articoli in corso e apprendimento dell'AI..."):  # Mostra lo spinner durante la ricerca
        
        docs=[]

        # Download with PyPaperBot

        # Download with PyPaperBot
        if platform.system()=='Windows': 
            subprocess.run(["powershell", "python", "-m", "PyPaperBot", "--query={}".format(topic.replace(" ", "+")), "--scholar-pages={}".format(pagine_scholar), "--scholar-results={}".format(n), "--min-year={}".format(year), "--dwn-dir={}".format(FOLDER_PATH), "--restrict=1"], shell=False)
        else:
            subprocess.run(["bash", "python", "-m", "PyPaperBot", "--query={}".format(topic.replace(" ", "+")), "--scholar-pages={}".format(pagine_scholar), "--scholar-results={}".format(n), "--min-year={}".format(year), "--dwn-dir={}".format(FOLDER_PATH), "--restrict=1"], shell=False)
        
        #Read files
        for pdf_file in os.listdir(FOLDER_PATH):
            if pdf_file.endswith('.pdf'):

                loader = PyPDFLoader(os.path.join(FOLDER_PATH, pdf_file))
                docs=docs+loader.load_and_split()

        #Split docs
        docs=text_splitter.fixed_split(docs)
        
        #Embed docs
        vectore_store.index_documents(docs)

        st.success("Trovati {} articoli relativi al tuo argomento".format(n))

def print_result(result):
  '''
  Print your results with Markup language
  '''
  output_text = f"""
  ### Answer: 
  {result['answer']}
  ### Sources: 
  {result['sources']}
  ### All relevant sources:
  {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """
  return(output_text)

def retrieve_chain():

    #create custom prompt for your use case
    prompt_template="""Sei Debbie, un'assistente alla ricerca AI che parla ai ricercatori universitari. 

    Rispondi alle domande utilizzando i fatti forniti. 
    Usa sempre cortesie e saluti gentili come "Buongiorno" e "Buon pomeriggio". 
    Utilizza i seguenti pezzi di contesto per rispondere alla domanda degli utenti.
    Prendi nota delle fonti e includile nella risposta nel formato: "SOURCES: source1 source2", usa "SOURCES" in maiuscolo indipendentemente dal numero di fonti.

    Traduci sempre tutto in italiano.

    Se non conosci la risposta, dì semplicemente "Non lo so", non cercare di inventare una risposta.
    ----------------
    {summaries}
    """

    messages = [
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain_type_kwargs = {"prompt": prompt}

    #build your chain for RAG+C
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

    return chain

###########################################################################

llm = ChatGroq(temperature=TEMPERATURE, model_name=MODEL_NAME)
chain = retrieve_chain()

st.title("Debbie - assistente alla ricerca")

col1, col2 = st.columns(2)
with col1:
    n = st.number_input("Numero di articoli da recuperare", min_value=1, max_value=10, value=3)
    year = st.number_input("Anno di pubblicazione", min_value=1900, max_value=2024, value=2023)
with col2:
    topic = st.text_input("Argomento trattato", "es. riabilitazione MSK")
    pagine_scholar = st.number_input("Numero di pagine da leggere", min_value=1, max_value=10, value=3)
    
# Aggiungi il pulsante alla tua app Streamlit
st.caption("Argomenti ricchi di informazioni possono richiedere tempi iniziali di ricerca degli articoli più lunghi .")
st.caption("Per questo motivo, è consigliato ricercare l'argomento selezionato tramite il pulsante sottostante prima di fare le tue domande .")
if st.button("Lancia la ricerca"):
    
    research_papers(topic, n, year, pagine_scholar)


# Chatbot ##################################################################

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL_NAME

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("La tua domanda?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        research_papers(topic, n, year, pagine_scholar)
                
        # #Create the stream 
        # stream = llm.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        # response = st.write_stream(stream)

        result = chain(prompt)
        response = st.markdown(print_result(result))
    
    st.session_state.messages.append({"role": "assistant", "content": print_result(result)})