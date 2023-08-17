# Open-AI-and-Azure-Cognitive-Search-with-LangChain
Open AI and Azure Cognitive Search with LangChain for Question Answering over Documents

Open AI and Azure Cognitive Search with LangChain for Question Answering over Documents
Our motive is to unleashing the Power of Open AI for building Question-Answering app over documents.
Question-Answering over Documents (QA over Documents) is a natural language processing (NLP) task that involves developing algorithms and models to automatically answer questions based on the content of a given document or set of documents.
Let me outline the steps involved in developing our desired app first and then dive into the code.
Azure Cognitive Search (formerly known as "Azure Search") is a cloud search service that gives developers infrastructure, APIs, and tools for building a rich search experience over private, heterogeneous content in web, mobile, and enterprise applications.
The first step in developing our app is to create index in the Azure Cognitive Search. Index is where we store the vector representations (embedding) for our data from the document.
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.indexes.models import (
    CorsOptions,
    SearchIndex
)

# Index details
# Give your index a name
index_name = "openai-search-demo"

# Search Index Schema definition
index_schema = "./openai-search-demo.json"

"""
Fields: 
    id
    content
    metadata
    category 
    content_vector : 
        # Manual Field Creation 
        # Collection(Edm.Single)
        # Retrievable, Searchable
        # dimension :: 1536
        # create a vector configuration with cosine similarity
        # kind : hnws
"""

## Class to create index ::
# Instantiate a client

    class CreateClient(object):
      def __init__(self, endpoint, key, index_name):
          self.endpoint = endpoint
          self.index_name = index_name
          self.key = key
          self.credentials = AzureKeyCredential(key)

    # Create a SearchClient
    # Use this to upload docs to the Index
    def create_search_client(self):
        return SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credentials,
        )

    # Create a SearchIndexClient
    # This is used to create, manage, and delete an index
    def create_admin_client(self):
        return SearchIndexClient(endpoint=endpoint, credential=self.credentials)

    start_client = CreateClient(endpoint, key, index_name)
    
    admin_client = start_client.create_admin_client()
    search_client = start_client.create_search_client()

# Get Schema from File or URL
    def get_schema_data(schema, url=False):
      if not url:
          with open(schema) as json_file:
              schema_data = json.load(json_file)
              return schema_data
      else:
          data_from_url = requests.get(schema)
          schema_data = json.loads(data_from_url.content)
          return schema_data
  
    schema_data = get_schema_data(index_schema, url=False)
  
    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    scoring_profiles = []
  
    index = SearchIndex(
          name=index_name,
          fields=schema_data["fields"],
          scoring_profiles=scoring_profiles,
          suggesters=schema_data["suggesters"],
          cors_options=cors_options,
        )
  
    upload_schema = admin_client.create_index(index)

Then we load our data from the document, We convert data into embedding (vector representation) using Azure Open AI model.
Azure Open AI is collection of large-scale, generative AI models with deep understandings of language and code to enable new reasoning and comprehension capabilities for building cutting-edge applications.

    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.llms import AzureOpenAI
    from langchain.text_splitter import CharacterTextSplitter
    
    pdf_file_path = 'PATH_TO_FILE'
    with open(pdf_folder_path, 'r+') as f:
        documents = f.readlines()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    llm = AzureOpenAI(deployment_name = deployment, 
                      model = model,
                      temperature=0.1)
    
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model,deployment=deployment,
                                       openai_api_base=openai.api_base,
                                      openai_api_type = "azure",
                                      chunk_size=1)
    
    for i,content in enumerate(texts):
        batch_array.append(
            {
                "id": str(i),
                "content": content,
                "metadata": str('{"id":' + str(i) +'}'),
                "category": "CATEGORY",
                "content_vector": embeddings.embed_query(content)
            })
            
Then we upload the vector data into our index. By this we create the Vector Search Database.
    results = search_client.upload_documents(documents=batch_array)

Vector search provides capability for indexing, storing, and retrieving vector embedding from a search index. We can use it to power similarity search, multi-modal search, recommendations engines, or applications implementing the Retrieval systems.
We can index vector data as fields in documents alongside textual and other types of content.

# Vector Store :: Azure Storage Account details:
    vector_store_address: str = os.getenv("AzureVectoStore")
    vector_store_password: str = os.getenv("AzureVectorPassword")
    
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        content_key="content"
    )

Now that we have done creating our vector database. We now create pipeline, where Azure Open AI model can query over vector representations, extracting relevant answers from it.

In simple terms, Azure Open AI model uses vector representations and fetch close match for the given user query.
We use Lang Chain for this. Lang Chain is a framework designed to simplify the creation of applications using large language models. It is a natural language processing library that provides various tools and models for working with text data.

Now create the Retrieval-QA chain with Open AI model. Remember to select return_source_documents so that we obtain the source documents from which our Open AI model fetches the answer.

    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    query = "What is IR35?"
    result = qa({"query": query})

# Done and Dusted !!
Our QA app is ready for providing answer to queries.

Here is the entire code. 
As a bonus, Azure Vector Search comes with filtering with index fields. So in order to filter our vector data fields use below code. 

    retriever.search_kwargs = {'filters': "search.ismatch('finance', 'category')"}
