from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

load_dotenv()

def extract_text(file_path):
    ext = file_path.lower().split('.')[-1]
    
    if ext == 'pdf':
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
    elif ext == 'docx':
        try:
            loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
            docs = loader.load()
        except:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['page'] = 0
    elif ext == 'txt':
        loader = TextLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['page'] = 0
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    for doc in docs:
        doc.page_content = doc.page_content
    return docs


def structured_data(data):
    prompt = f"""
    use the shipment data: {data}

    {{
    "Shipment_id": "",
    "shipper": "",
    "consignee": "",
    "pickup_datetime": "",
    "delivery_datetime": "",
    "equipment_type": "",
    "mode": "",
    "rate": "",
    "currency": "",
    "weight": "",
    "carrier_name": ""
    }}

    Return JSON with nulls if missing.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides structured data."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    optimized_json = response.choices[0].message.content
    optimized_json = json.loads(optimized_json)
    return optimized_json


    


def build_rag(filepath):
    docs = extract_text(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the provided context only. If the Question is not related to the Context, say "Answer not found."

    Context: {context}

    Question: {input}

    Answer:""")
    
    llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain, vectorstore


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

rag_chain = None
vectorstore = None
current_filepath = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global rag_chain, vectorstore
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        global current_filepath
        rag_chain, vectorstore = build_rag(filepath)
        current_filepath = filepath
        return jsonify({'message': 'File uploaded and processed successfully', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    if not rag_chain or not vectorstore:
        return jsonify({'error': 'No document uploaded yet'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=2)
        best_doc, best_score = max(results, key=lambda x: x[1])
        
        if best_score > 0:
            response = rag_chain.invoke({"input": question})
            return jsonify({
                'answer': response['answer'],
                'score': float(round(best_score, 3)),
                'source': best_doc.page_content[:100] + '...'
            })
        else:
            return jsonify({'answer': 'Answer not found.', 'score': float(round(best_score, 3)), 'source': ''})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract():
    if not current_filepath:
        return jsonify({'error': 'No document uploaded yet'}), 400
    
    try:
        docs = extract_text(current_filepath)
        full_text = " ".join([doc.page_content for doc in docs])
        structured = structured_data(full_text)
        return jsonify(structured)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
