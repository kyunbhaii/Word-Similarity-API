# 🔍 Word Similarity API

A FastAPI-based REST API that returns semantically similar words using a custom-trained Word2Vec model on Stack Overflow technical Q&A data.

---

## 📁 Project Structure

```
├── QuerySim.ipynb
├── app.py
├── word2vec_model.pkl
├── stack_overflow_tech_final.parquet
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- Python ≥ 3.8
- pip package manager

### 2️⃣ Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

**Required packages:**
```
fastapi
uvicorn
pydantic
gensim
pandas
numpy
nltk
```

### 3️⃣ Running the API

Start the server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📡 API Endpoints

### **GET /** - Home
```bash
curl http://localhost:8000/
```
**Response:**
```json
{
  "message": "Word Similarity API is running!"
}
```

### **GET /similar/{word}** - Get Similar Words
```bash
curl "http://localhost:8000/similar/python?topn=5"
```

**Parameters:**
| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|-------|
| `word` | string | Query word (path parameter) | - | - |
| `topn` | integer | Number of similar words to return | 5 | 1-25 |

**Response:**
```json
{
  "query": "python",
  "topn": 5,
  "results": [
    {"word": "django", "score": 0.8521},
    {"word": "flask", "score": 0.8234},
    {"word": "pip", "score": 0.8012},
    {"word": "virtualenv", "score": 0.7845},
    {"word": "numpy", "score": 0.7623}
  ]
}
```

**Error Responses:**
- `400` - Empty word provided
- `404` - Word not found in vocabulary

### **GET /health** - Health Check
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## 🧠 Model Details

### Training Process

The Word2Vec model was trained on **3,296 Stack Overflow answers** from Python-related questions.

**Training Parameters:**
- **Architecture:** Skip-gram (sg=1)
- **Vector size:** 200 dimensions
- **Context window:** 5 words
- **Min word frequency:** 5 occurrences
- **Epochs:** 10
- **Workers:** 4 (parallel processing)

**Model Statistics:**
- **Vocabulary size:** 4,492 unique words
- **Training corpus:** 208,557 total words
- **Average tokens per answer:** 63.3

**Top technical terms learned:**
`python`, `import`, `use`, `def`, `return`, `print`, `class`, `function`, `file`, `list`, `install`, `object`, `example`

---

## 📊 Dataset

**Source:** Stack Overflow technical Q&A  
**Format:** Parquet file  
**Size:** 3,296 rows  
**Columns:**
- `title` - Question title
- `question` - Full question text
- `answer` - Answer text (used for training)

---

## 🔧 Text Preprocessing Pipeline

The following preprocessing steps were applied to clean the answer text:

1. **Tech word normalization** - Standardized technology terms:
   - `C++` → `cpp`
   - `C#` → `csharp`
   - `.NET` → `dotnet`
   - `Node.js` → `nodejs`
   - `scikit-learn` → `sklearn`
   - And 20+ more tech terms

2. **URL removal** - Stripped all HTTP/HTTPS links

3. **HTML tag removal** - Cleaned HTML markup

4. **Emoji removal** - Removed non-ASCII characters

5. **Number removal** - Eliminated numeric tokens

6. **Punctuation removal** - Stripped all punctuation

7. **Lowercasing** - Converted all text to lowercase

8. **Stopword removal** - Removed common English stopwords using NLTK

9. **Extra space removal** - Normalized whitespace

**Result:** Clean, tokenized text optimized for Word2Vec training

---

## 🎯 Example Use Cases

### Find similar programming languages:
```bash
curl "http://localhost:8000/similar/javascript?topn=3"
```

### Discover related libraries:
```bash
curl "http://localhost:8000/similar/numpy?topn=5"
```

### Explore framework alternatives:
```bash
curl "http://localhost:8000/similar/django?topn=10"
```

---

## 📈 Model Performance

**Vocabulary Statistics:**
- **Total unique words in corpus:** 24,469
- **Model vocabulary:** 4,492 words (filtered by min_count=5)
- **Filtered out:** 19,977 rare words (appearing <5 times)

**Sample Similarity Results:**

Query: `http`
```
interact (0.91), restful (0.89), mongodb (0.88), 
applications (0.87), web (0.87)
```

Query: `class`
```
myclass (0.78), setattr (0.77), inheriting (0.77), 
inherit (0.75), upperattrmetaclasstype (0.75)
```

Query: `sql`
```
mysql (0.82), cursor (0.81), domains (0.80), 
entirely (0.80), oursql (0.80)
```

---

## 🔬 Advanced Features

### 1. **Python-Focused Vocabulary**
The model captures relationships within Python programming concepts:
- Object-oriented programming (class, inherit, setattr, myclass)
- Database operations (sql, mysql, cursor)
- Web protocols (http, restful, web, applications)
- Common programming patterns found in Stack Overflow answers

**Note:** Due to the focused dataset and min_count=5 filtering, the vocabulary is limited to frequently discussed Python-related terms. Many general tech terms may not be present in the model.

### 2. **Context-Based Similarity**
Uses Skip-gram architecture to capture:
- Syntactic patterns
- Semantic relationships
- Co-occurrence patterns in technical discussions

### 3. **Efficient Inference**
- Model loaded once at startup
- Fast vector lookups using Gensim
- Normalized similarity scores (0-1 range)

---

## 🛠️ Retraining the Model

To retrain with your own data:

1. Open `QuerySim.ipynb` in Jupyter/Colab
2. Replace the dataset with your own text corpus
3. Adjust preprocessing as needed
4. Modify Word2Vec hyperparameters:
   ```python
   model = Word2Vec(
       sentences=tokenised_words,
       vector_size=200,    # Embedding dimension
       window=5,           # Context window size
       min_count=5,        # Minimum word frequency
       workers=4,          # CPU cores
       sg=1,               # 1=Skip-gram, 0=CBOW
       epochs=10           # Training iterations
   )
   ```
5. Save the new model:
   ```python
   with open('word2vec_model.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

---

## 🐛 Troubleshooting

### Issue: Model file not found
**Error:** `FileNotFoundError: word2vec_model.pkl`

**Solution:** Ensure `word2vec_model.pkl` is in the same directory as `app.py`

### Issue: Word not in vocabulary
**Error:** `404: 'xyz' not found in vocabulary`

**Solution:** The word doesn't appear ≥5 times in training data. Try related terms or retrain with lower `min_count`.

### Issue: Slow startup
**Cause:** Large model file being loaded

**Solution:** This is normal. The model loads once at startup (~2-3 seconds).

---

## 📝 Notes

- Model was trained on **Python-focused** Stack Overflow data, so it performs best with technical programming terms
- The API is case-insensitive (all queries are lowercased)
- Words must have appeared at least 5 times in the training corpus to be in vocabulary
- Similarity scores are cosine similarities ranging from -1.0 (completely dissimilar) to 1.0 (identical), though in practice the API returns mostly positive values (0 to 1) for "most similar" queries

---

## 🤝 Contributing

To improve the model:
1. Add more diverse technical Q&A data
2. Experiment with different preprocessing strategies
3. Tune Word2Vec hyperparameters (window size, vector dimensions)
4. Try alternative architectures (FastText, GloVe)

---

## 📄 License

This project is licensed under the MIT License.

Free for educational and commercial use with attribution.

---

## 🔗 Related Resources

- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Word Embeddings Explained](https://arxiv.org/abs/1301.3781)
