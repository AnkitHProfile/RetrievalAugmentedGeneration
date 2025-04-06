# 🤖 Retrieval-Augmented Generation (RAG) Based Question-Answering System

This project demonstrates how to build a **Question-Answering System** using the **Retrieval-Augmented Generation (RAG)** technique with Hugging Face Transformers, FAISS, and Google Colab. The system retrieves the most relevant content from a custom text dataset and uses a pretrained model to generate accurate, context-aware answers.

---

## 🧠 Project Objective

To answer user queries accurately by combining:
- The query itself,
- Relevant context retrieved from custom documents,
- And the pretrained knowledge of a transformer-based language model (RAG).

---

## 🚀 Key Features

- 🔍 **Document Retrieval using FAISS** and Dense Passage Retrieval (DPR)
- 🧾 **Custom Document Ingestion** from Google Drive (.txt files)
- 🧠 **Transformer-based Question Answering** using Hugging Face’s RAG model
- ⚙️ **GPU Support** via Google Colab
- 📄 **Structured Dataset Creation** using `datasets` and `pandas`
- 📥 **Embeddings Indexed and Searched** via FAISS (HNSW + MIPS)

---

## 🛠️ Tech Stack

| Component      | Technology                                     |
|----------------|------------------------------------------------|
| Language       | Python                                         |
| Libraries      | Transformers, FAISS, PyTorch, Pandas, NumPy    |
| Embedding      | DPRQuestionEncoder + FAISS Index               |
| Retrieval      | HNSW-based Similarity Search                   |
| Generation     | Hugging Face RAG Model                         |
| Environment    | Google Colab                                   |

---

## 📦 Setup Instructions (Google Colab)

1. **Install Dependencies**
   ```python
   !pip install faiss-cpu datasets transformers
   ```

2. **Mount Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Fetch Text Files**
   - Store your `.txt` files in a folder named `Files/` inside Google Drive.
   - The code reads and processes all `.txt` files, extracting **titles** and **articles**.

4. **Generate and Store Embeddings**
   - Use BERT (or DPR) to generate embeddings for document chunks.
   - Store them in a FAISS index for similarity search.

5. **Set Up the Retriever + RAG Pipeline**
   - Use `RagRetriever` from Hugging Face to retrieve top documents based on a query.
   - Use `RagTokenForGeneration` to generate final answers.

6. **Ask Questions**
   - Use the `ask_question()` function to get answers:
     ```python
     ask_question("What is Hierarchical RAG?")
     ```

---

## 📊 How It Works

1. **Preprocessing**
   - Reads all `.txt` files
   - Splits articles into chunks
   - Generates embeddings via DPR

2. **FAISS Index**
   - All embeddings are indexed using FAISS (HNSW + MIPS)

3. **Retrieval**
   - For a given query, 3 most similar document chunks are fetched

4. **Answer Generation**
   - Uses RAG to generate answers based on the retrieved context

---

## 📁 Project Structure

```
RAG_QA_System/
├── RAG.ipynb                 # Main Colab Notebook
├── Retrieval Augmented Generation (RAG).pdf  # Explanation of process
├── Files/                    # Folder in Google Drive containing .txt files
```

---

## 🧪 Example Questions

```python
ask_question("How many models have been created by Cerebras?")
ask_question("What is Hierarchical RAG?")
```

The model accurately retrieves and answers based on both custom documents and pretrained knowledge.

---

## ✅ Conclusion

This project showcases how to combine document retrieval and generation techniques using modern NLP libraries to create a powerful, context-aware Q&A system. The use of FAISS indexing and Hugging Face's RAG model allows for scalable, intelligent information retrieval over large text corpora.

---

## 👨‍💻 Contributor

**Ankit Hiremath**  
GitHub: [AnkitHProfile](https://github.com/AnkitHProfile)
