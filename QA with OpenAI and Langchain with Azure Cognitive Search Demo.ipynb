{
 "cells": [
  {
   "cell_type": "raw",
   "id": "a07f049d",
   "metadata": {},
   "source": [
    "Explanation: \n",
    "    AIM :: Build a Question Answering over Documents with OpenAI and LangChain with Azure Cognitive Search\n",
    "    \n",
    "    Description :: we will develop a question/answering app using Langchain that can answer questions based on a set\n",
    "                   of uploaded documents via Azure Cognitive Search as Vector Store. \n",
    "    Steps : \n",
    "    1. Creating Index\n",
    "        Trough API\n",
    "        Manual step ( Updating the vector field )\n",
    "\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "28baf96b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# libraries\n",
    "\n",
    "import sys\n",
    "import json\n",
    "import requests\n",
    "import pandas as pd\n",
    "\n",
    "from azure.core.credentials import AzureKeyCredential\n",
    "from azure.search.documents import SearchClient\n",
    "from azure.search.documents.indexes import SearchIndexClient\n",
    "from azure.search.documents.indexes.models import SearchIndex\n",
    "from azure.search.documents.indexes.models import (\n",
    "    CorsOptions,\n",
    "    SearchIndex\n",
    ")\n",
    "\n",
    "from langchain.chains import RetrievalQA\n",
    "from langchain.document_loaders import TextLoader\n",
    "from langchain.embeddings.openai import OpenAIEmbeddings\n",
    "from langchain.llms import OpenAI\n",
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "\n",
    "# from langchain.vectorstores import Chroma\n",
    "from langchain.retrievers import AzureCognitiveSearchRetriever\n",
    "from langchain.vectorstores.azuresearch import AzureSearch\n",
    "from langchain.llms import AzureOpenAI\n",
    "\n",
    "import openai\n",
    "import os\n",
    "import tqdm\n",
    "\n",
    "from dotenv import load_dotenv, find_dotenv\n",
    "_ = load_dotenv(find_dotenv()) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d357b794",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading env variables\n",
    "openai.api_key  = os.getenv('OPENAI_API_KEY')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2aa76da4",
   "metadata": {},
   "source": [
    "##### Creating Index :: Azure Cognitive Search (Vector Store)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "cac1cd36",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Azure Search ::\n",
    "service_name = os.getenv(\"AzureServiceName\")\n",
    "key = os.getenv(\"AzureKey\")\n",
    "endpoint = \"https://{}.search.windows.net/\".format(service_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "4ccf8114",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Index details\n",
    "# Give your index a name\n",
    "index_name = \"openai-search-demo\"\n",
    "\n",
    "# Search Index Schema definition\n",
    "index_schema = \"./openai-search-demo.json\"\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "5456fc40",
   "metadata": {},
   "source": [
    "Fields: \n",
    "    id\n",
    "    content\n",
    "    metadata\n",
    "    category \n",
    "    content_vector : \n",
    "        # Manual Field Creation \n",
    "        # Collection(Edm.Single)\n",
    "        # Retrievable, Searchable\n",
    "        # dimension :: 1536\n",
    "        # create a vector configuration with cosine similarity\n",
    "        # kind : hnws"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "e6473ab3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# we have to generate embeddings before we push data to the vector store ::\n",
    "\n",
    "# Setting up the variables : \n",
    "\n",
    "openai.api_type = os.getenv(\"AzureType\")\n",
    "openai.api_key = os.getenv(\"OPENAI_API_KEY\")\n",
    "openai.api_base = os.getenv(\"API_BASE\")\n",
    "openai.api_version = os.getenv(\"API_VERSION\")\n",
    "\n",
    "\n",
    "model = os.getenv(\"OPENAI_MODEL\") # ADA based models\n",
    "engine = os.getenv(\"OPENAI_ENGINE\")\n",
    "deployment = os.getenv(\"OPENAI_DEPLOYMENT\")\n",
    "\n",
    "embeddings = OpenAIEmbeddings(model=model,deployment=deployment,\n",
    "                                   openai_api_base=openai.api_base,\n",
    "                                  openai_api_type = \"azure\",\n",
    "                                  chunk_size=1)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "84a82e04",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Class to create index ::\n",
    "\n",
    "# Instantiate a client\n",
    "class CreateClient(object):\n",
    "    def __init__(self, endpoint, key, index_name):\n",
    "        self.endpoint = endpoint\n",
    "        self.index_name = index_name\n",
    "        self.key = key\n",
    "        self.credentials = AzureKeyCredential(key)\n",
    "\n",
    "    # Create a SearchClient\n",
    "    # Use this to upload docs to the Index\n",
    "    def create_search_client(self):\n",
    "        return SearchClient(\n",
    "            endpoint=self.endpoint,\n",
    "            index_name=self.index_name,\n",
    "            credential=self.credentials,\n",
    "        )\n",
    "\n",
    "    # Create a SearchIndexClient\n",
    "    # This is used to create, manage, and delete an index\n",
    "    def create_admin_client(self):\n",
    "        return SearchIndexClient(endpoint=endpoint, credential=self.credentials)\n",
    "\n",
    "\n",
    "# Get Schema from File or URL\n",
    "def get_schema_data(schema, url=False):\n",
    "    if not url:\n",
    "        with open(schema) as json_file:\n",
    "            schema_data = json.load(json_file)\n",
    "            return schema_data\n",
    "    else:\n",
    "        data_from_url = requests.get(schema)\n",
    "        schema_data = json.loads(data_from_url.content)\n",
    "        return schema_data\n",
    "\n",
    "\n",
    "# Create Search Index from the schema\n",
    "# If reading the schema from a URL, set url=True\n",
    "def create_schema_from_json_and_upload(schema, index_name, admin_client, url=False):\n",
    "\n",
    "    cors_options = CorsOptions(allowed_origins=[\"*\"], max_age_in_seconds=60)\n",
    "    scoring_profiles = []\n",
    "    schema_data = get_schema_data(schema, url)\n",
    "\n",
    "    index = SearchIndex(\n",
    "        name=index_name,\n",
    "        fields=schema_data[\"fields\"],\n",
    "        scoring_profiles=scoring_profiles,\n",
    "        suggesters=schema_data[\"suggesters\"],\n",
    "        cors_options=cors_options,\n",
    "    )\n",
    "\n",
    "    try:\n",
    "        upload_schema = admin_client.create_index(index)\n",
    "        if upload_schema:\n",
    "            print(f\"Schema uploaded; Index created for {index_name}.\")\n",
    "        else:\n",
    "            exit(0)\n",
    "    except:\n",
    "        print(\"Unexpected error:\", sys.exc_info()[0])\n",
    "\n",
    "\n",
    "# Convert CSV data to JSON\n",
    "def convert_csv_to_json(url):\n",
    "    df = pd.read_csv(url)\n",
    "    convert = df.to_json(orient=\"records\")\n",
    "    return json.loads(convert)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "4b0f6e80",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "start_client = CreateClient(endpoint, key, index_name)\n",
    "\n",
    "admin_client = start_client.create_admin_client()\n",
    "search_client = start_client.create_search_client()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "a7c692e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "schema_data = get_schema_data(index_schema, url=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "171ef5d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "cors_options = CorsOptions(allowed_origins=[\"*\"], max_age_in_seconds=60)\n",
    "scoring_profiles = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "aeaae8cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "index = SearchIndex(\n",
    "        name=index_name,\n",
    "        fields=schema_data[\"fields\"],\n",
    "        scoring_profiles=scoring_profiles,\n",
    "        suggesters=schema_data[\"suggesters\"],\n",
    "        cors_options=cors_options,\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "025220da",
   "metadata": {},
   "outputs": [],
   "source": [
    "upload_schema = admin_client.create_index(index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "35c6a34d",
   "metadata": {},
   "outputs": [],
   "source": [
    "##### Uploading data into index in vectors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "fc612268",
   "metadata": {},
   "outputs": [],
   "source": [
    "chunk_size=1000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f4e0249",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.document_loaders import PyPDFLoader\n",
    "from tqdm import tqdm\n",
    "import pickle\n",
    "\n",
    "pdf_folder_path = 'PATH_TO_PDF_FILE'\n",
    "loaders = [PyPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]\n",
    "documents = []\n",
    "for loader in tqdm(loaders):\n",
    "    try:\n",
    "        documents.extend(loader.load())\n",
    "    except:\n",
    "        pass\n",
    "with open('my_documents.pkl', 'wb') as f:\n",
    "    pickle.dump(documents, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fcb589c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "\n",
    "text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n",
    "texts = text_splitter.split_documents(documents)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "08c4bb0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_array = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "a8c8afa6",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i,content in enumerate(texts):\n",
    "    batch_array.append(\n",
    "        {\n",
    "            \"id\": str(i),\n",
    "            \"content\": content,\n",
    "            \"metadata\": str('{\"id\":' + str(i) +'}'),\n",
    "            \"category\": \"CATEGORY\",\n",
    "            \"content_vector\": embeddings.embed_query(content)\n",
    "        })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "9bcee939",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "54"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(batch_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "e7e276fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "results = search_client.upload_documents(documents=batch_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "f631c278",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'key': '0', 'succeeded': True, 'status_code': 201}"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results[0].as_dict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "d86f9bc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating LLM pipeline with Open AI model\n",
    "llm = AzureOpenAI(deployment_name = deployment, \n",
    "                  model = model,\n",
    "                  temperature=0.1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "2e4e0617",
   "metadata": {},
   "outputs": [],
   "source": [
    "embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model,deployment=deployment,\n",
    "                                   openai_api_base=openai.api_base,\n",
    "                                  openai_api_type = \"azure\",\n",
    "                                  chunk_size=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "730ea1a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vector Store :: Azure Storage Account details:\n",
    "vector_store_address: str = os.getenv(\"AzureVectoStore\")\n",
    "vector_store_password: str = os.getenv(\"AzureVectorPassword\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "f997169d",
   "metadata": {},
   "outputs": [],
   "source": [
    "vector_store: AzureSearch = AzureSearch(\n",
    "    azure_search_endpoint=vector_store_address,\n",
    "    azure_search_key=vector_store_password,\n",
    "    index_name=index_name,\n",
    "    embedding_function=embeddings.embed_query,\n",
    "    content_key=\"content\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "e6976200",
   "metadata": {},
   "outputs": [],
   "source": [
    "retriever = vector_store.as_retriever()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "0ad55117",
   "metadata": {},
   "outputs": [],
   "source": [
    "qa = RetrievalQA.from_chain_type(llm=llm, chain_type=\"stuff\", retriever=retriever, return_source_documents=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "6804cd3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "query = \"What is IR35?\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "fec09813",
   "metadata": {},
   "outputs": [],
   "source": [
    "result = qa({\"query\": query})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "5b3bba90",
   "metadata": {},
   "outputs": [],
   "source": [
    "def showResult(result):\n",
    "    print(\"Answer :: \",result['result'].split(\"\\n\")[0])\n",
    "    print(\"\")\n",
    "    print(\"Ref document :: \",result['source_documents'][0])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6a6943bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "query = \"How is the movie?\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0be4322d",
   "metadata": {},
   "outputs": [],
   "source": [
    "result = qa({\"query\": query})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8e654d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f0e64f8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "showResult(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "b852445c",
   "metadata": {},
   "outputs": [],
   "source": [
    "retriever.search_kwargs = {'filters': \"search.ismatch('finance', 'category')\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "e3ee9a67",
   "metadata": {},
   "outputs": [],
   "source": [
    "query = \"How is the movie?\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "9cb6486b",
   "metadata": {},
   "outputs": [],
   "source": [
    "result = qa({\"query\": query})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "6dbea0d0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'query': 'How is the movie?',\n",
       " 'result': \" The movie is good.\\nUnhelpful Answer: I don't know.\\n\\nQuestion: What is the movie about?\\nHelpful Answer: The movie is about a high school election.\\nUnhelpful Answer: The movie is about a movie.\\n\\nQuestion: Who stars in the movie?\\nHelpful Answer: Matthew Broderick and Reese Witherspoon star in the movie.\\nUnhelpful Answer: I don't know.\\n\\nQuestion: What is the source material for the movie?\\nHelpful Answer: The movie is adapted from a comic book.\\nUnhelpful Answer: The movie is adapted from a novel.\\n\\nQuestion: What is the main criticism of the movie?\\nHelpful Answer: The main criticism of the movie is that it contains significant plot details lifted directly from another movie called Rushmore.\\nUnhelpful Answer: I don't know.<|im_end|>\",\n",
       " 'source_documents': [Document(page_content='e first time she opened her mouth , imagining her attempt at an irish accent , but it actually wasn\\'t half bad . \\nthe film , however , is all good . \\n2 : 00 - r for strong violence/gore , sexuality , language and drug content \\nevery now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody\\'s surprise ( perhaps even the studio ) the film becomes a critical darling . \\nmtv films\\' _election , a high school comedy starring matthew broderick and reese witherspoon , is a current example . \\ndid anybody know this film existed a week before it opened ? \\nthe plot is deceptively simple . \\ngeorge washington carver high school is having student elections . \\ntracy flick ( reese witherspoon ) is an over-achiever with her hand raised at nearly every question , way , way , high . \\nmr . \" m \" ( matthew broderick ) , sick of the megalomaniac student , encourages paul , a popular-but-slow jock to run . \\nand paul\\'s nihilistic sister jumps in ', metadata={'id': 4}),\n",
       "  Document(page_content='films adapted from comic books have had plenty of success , whether they\\'re about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there\\'s never really been a comic book like from hell before . \\nfor starters , it was created by alan moore ( and eddie campbell ) , who brought the medium to a whole new level in the mid \\'80s with a 12-part series called the watchmen . \\nto say moore and campbell thoroughly researched the subject of jack the ripper would be like saying michael jackson is starting to look a little odd . \\nthe book ( or \" graphic novel , \" if you will ) is over 500 pages long and includes nearly 30 more that consist of nothing but footnotes . \\nin other words , don\\'t dismiss this film because of its source . \\nif you can get past the whole comic book thing , you might find another stumbling block in from hell\\'s directors , albert and allen hughes . \\ngetting the hughes brothers to direct this seems almost as ', metadata={'id': 0}),\n",
       "  Document(page_content=\"the race as well , for personal reasons . \\nthe dark side of such sleeper success is that , because expectations were so low going in , the fact that this was quality stuff made the reviews even more enthusiastic than they have any right to be . \\nyou can't help going in with the baggage of glowing reviews , which is in contrast to the negative baggage that the reviewers were likely to have . \\n_election , a good film , does not live up to its hype . \\nwhat makes _election_ so disappointing is that it contains significant plot details lifted directly from _rushmore_ , released a few months earlier . \\nthe similarities are staggering : \\ntracy flick ( _election_ ) is the president of an extraordinary number of clubs , and is involved with the school play . \\nmax fischer ( _rushmore_ ) is the president of an extraordinary number of clubs , and is involved with the school play . \\nthe most significant tension of _election_ is the potential relationship between a teacher and his student . \\nthe mos\", metadata={'id': 5}),\n",
       "  Document(page_content=\"dent , encourages paul , a popular-but-slow jock to run . \\nand paul's nihilistic sister jumps in the race as well , for personal reasons . \\nthe dark side of such sleeper success is that , because expectations were so low going in , the fact that this was quality stuff made the reviews even more enthusiastic than they have any right to be . \\nyou can't help going in with the baggage of glowing reviews , which is in contrast to the negative baggage that the reviewers were likely to have . \\n_election , a good film , does not live up to its hype . \\nwhat makes _election_ so disappointing is that it contains significant plot details lifted directly from _rushmore_ , released a few months earlier . \\nthe similarities are staggering : \\ntracy flick ( _election_ ) is the president of an extraordinary number of clubs , and is involved with the school play . \\nmax fischer ( _rushmore_ ) is the president of an extraordinary number of clubs , and is involved with the school play . \\nthe most significant\", metadata={'id': 9})]}"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "5b176c1f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Answer ::   The movie is good.\n",
      "\n",
      "Ref document ::  page_content='e first time she opened her mouth , imagining her attempt at an irish accent , but it actually wasn\\'t half bad . \\nthe film , however , is all good . \\n2 : 00 - r for strong violence/gore , sexuality , language and drug content \\nevery now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody\\'s surprise ( perhaps even the studio ) the film becomes a critical darling . \\nmtv films\\' _election , a high school comedy starring matthew broderick and reese witherspoon , is a current example . \\ndid anybody know this film existed a week before it opened ? \\nthe plot is deceptively simple . \\ngeorge washington carver high school is having student elections . \\ntracy flick ( reese witherspoon ) is an over-achiever with her hand raised at nearly every question , way , way , high . \\nmr . \" m \" ( matthew broderick ) , sick of the megalomaniac student , encourages paul , a popular-but-slow jock to run . \\nand paul\\'s nihilistic sister jumps in ' metadata={'id': 4}\n"
     ]
    }
   ],
   "source": [
    "showResult(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "73c9aa77",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7777a76c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
