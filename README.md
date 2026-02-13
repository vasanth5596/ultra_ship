# RAG Document Assistant

A Flask-based web application that lets you upload documents and interact with them through Q&A and structured data extraction.

## What This Does

Upload a PDF, DOCX, or TXT file, then:

- Ask questions about the content
- Extract structured shipment data automatically

## Architecture

Pretty straightforward setup:

**Frontend**: Single HTML page with vanilla JavaScript. No frameworks, just simple fetch API calls to the backend.

**Backend**: Flask server with three endpoints:

- `/upload` - Takes your file, chunks it, creates embeddings, stores in FAISS
- `/ask` - Searches the vector store, retrieves relevant chunks, sends to LLM for answer
- `/extract` - Grabs full document text, sends to GPT-4 for structured JSON extraction

**Storage**: FAISS vector database runs in-memory. When you upload a new file, it overwrites the previous one. No persistence between server restarts.

**Models**:

- OpenAI `text-embedding-3-large` with 32 dimensions (small footprint)
- `gpt-4o-mini` for Q&A
- `gpt-4.1` for structured extraction

## Chunking Strategy

Using `RecursiveCharacterTextSplitter` with:

- Chunk size: 1000 characters
- Overlap: 100 characters

Why these numbers? 1000 chars is roughly 200-250 tokens, which keeps chunks small enough to be specific but large enough to maintain context. The 100-char overlap helps when important info sits at chunk boundaries.

The recursive splitter tries to break on natural boundaries (paragraphs, sentences) rather than mid-word, which keeps semantic meaning intact.

## Retrieval Method

Basic similarity search using FAISS with cosine similarity. We pull the top 2 most relevant chunks (k=2) for each query.

The embedding model is `text-embedding-3-large` but compressed to 32 dimensions instead of the default 3072. This trades some accuracy for speed and memory. For shipment documents, the loss is minimal since the vocabulary is pretty domain-specific.

## Guardrails

Two main safety checks:

1. **Context relevance check**: Before answering, we check if similarity scores are > 0. If not, return "Answer not found" instead of hallucinating.
2. **Prompt constraint**: The system prompt explicitly tells the LLM to only answer from provided context. If info isn't there, say so.

That's it. No fancy content filtering or input sanitization beyond what Flask provides. In production, you'd want to add rate limiting, input validation, and probably some content moderation.

## Confidence Scoring

We use FAISS similarity scores. Higher = better match.

The scores print to console during `/ask` requests so you can see what's happening. Right now there's just a binary check (score > 0), but you could easily add thresholds:

- Score > 0.5: High confidence
- Score 0.4-0.7: Medium confidence
- Score < 0: Low confidence, maybe don't answer

The current implementation doesn't expose scores to the user, which is probably a mistake. Users should know when the system is guessing.

## Known Failure Cases

1. **Multi-page context**: If an answer requires info from multiple pages that don't end up in the same chunks, the system will miss it. The k=2 retrieval is pretty limiting.
2. **Numerical reasoning**: Ask "what's the total weight of all shipments?" and it'll fail. The system can't do math across chunks.
3. **Ambiguous questions**: "What's the date?" - which date? Pickup? Delivery? The LLM will guess or say it can't answer.
4. **Document format issues**: Tables in PDFs often get mangled during extraction. The structured extraction might miss fields.
5. **Case sensitivity**: Everything gets lowercased during extraction, which breaks proper nouns and can confuse the LLM.
6. **Single document limit**: Upload a new file and the old one is gone. No multi-document search.
7. **No conversation memory**: Each question is independent. Can't do follow-ups like "what about the other one?"

## Ideas for Improvement

**Short term (easy wins)**:

- Increase k to 5 or 10 for retrieval
- Remove the lowercasing - it's causing more harm than good
- Add confidence scores to the UI
- Store multiple documents instead of overwriting
- Add file type validation on backend
- Implement proper error messages instead of generic 500s
- we can wevaitae for faster retirval

**Medium term**:

- Add conversation history so users can ask follow-up questions
- Implement hybrid search (keyword + semantic)
- Add document metadata filtering (search by date, type, etc.)
- Better table extraction for PDFs
- Add streaming responses so users see answers as they generate
- Implement proper logging and monitoring

**Long term (bigger changes)**:

- Switch to a persistent vector DB (Pinecone, Weaviate, etc.)
- Add multi-document reasoning
- Implement query rewriting to handle ambiguous questions
- Add citation/source tracking (which chunk/page the answer came from)
- Fine-tune embeddings on domain-specific data
- Add OCR for scanned documents
- Implement user authentication and document access control
- Add batch processing for multiple files
- Create an evaluation dataset to measure accuracy over time

**The structured extraction endpoint** is actually pretty solid. Sending the full text to GPT-4 works better than trying to extract from chunks. The main issue is cost - full documents can be expensive. Could optimize by:

- Using a smaller model for simple extractions
- Caching results
- Only sending relevant sections if you can identify them first

## Running It

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`

You'll need an OpenAI API key :

## enter key--> choose file--> upload--> ask question

## Tech Stack

- Flask (backend)
- LangChain (RAG pipeline)
- FAISS (vector store)
- OpenAI (embeddings + LLM)
- PyMuPDF, docx2txt, unstructured (document loaders)

No database, no auth, no deployment config. This is a prototype.
