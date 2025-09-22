from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os
import re
import json
import nltk
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

nltk.download("punkt")
PDF_FOLDER = "PDFs/"
MILVUS_HOST = "0.0.0.0"
MILVUS_PORT = "19530"
COLLECTION_NAME = "MilvusCollection"
embedding_model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
EMBEDDING_DIM = 768


def TextClean(text):
    return re.sub(r"\s{3,}", " ", text).strip().lower()


def get_embedding(texts):
    try:
        cleaned_texts = [TextClean(t) for t in texts]
        return embedding_model.encode(cleaned_texts, convert_to_numpy=True).tolist()
    except Exception as e:
        print(f"Embedding hatası: {e}")
        return None


def createIndex(collection):
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": 32,
            "efConstruction": 500,
        }
    }
    collection.createIndex(field_name="embedding", index_params=index_params)
    print("İndeks başarıyla oluşturuldu.")


def createMilvusColc():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"Milvus bağlantı hatası: {e}")
        raise

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"{COLLECTION_NAME} koleksiyonu silindi.")

    print(f"{COLLECTION_NAME} koleksiyonu oluşturuluyor...")
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema = CollectionSchema(fields=[id_field, embedding_field])
    collection = Collection(COLLECTION_NAME, schema)

    createIndex(collection)

    try:
        collection.load()
        print(f"{COLLECTION_NAME} koleksiyonu başarıyla yüklendi.")
    except Exception as e:
        print(f"Koleksiyon yükleme hatası: {e}")
        raise

    return collection


def load_chunks_from_pdfs(pdf_folder):
    all_chunks = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text = text.strip().lower()
                    if text:
                        all_chunks.append(text)
        except Exception as e:
            print(f"PDF dosyası okunamadı -> {pdf_path} - {e}")
            continue
    return all_chunks


def set_vecDB():
    collection = createMilvusColc()
    all_chunks = load_chunks_from_pdfs(PDF_FOLDER)
    print(f"Toplam {len(all_chunks)} chunk yüklendi.")

    current_id = 0
    batch_size = 100

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        embeddings = get_embedding(batch)
        if embeddings:
            ids = list(range(current_id, current_id + len(embeddings)))
            try:
                collection.insert([ids, embeddings])
                print(f"{len(embeddings)} embedding başarıyla eklendi. (ID aralığı: {current_id}-{current_id + len(embeddings) - 1})")
                current_id += len(embeddings)
            except Exception as e:
                print(f"Ekleme hatası: {e}")
                continue

    try:
        with open("chunk_texts.json", "w", encoding="utf-8") as f:
            all_data = [{"id": idx, "text": chunk} for idx, chunk in enumerate(all_chunks)]
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        print("chunk_texts.json başarıyla kaydedildi.")
    except Exception as e:
        print(f"JSON kaydetme hatası: {e}")

    print("Vektör veritabanı başarıyla oluşturuldu.")


if __name__ == "__main__":
    try:
        set_vecDB()
    except Exception as e:
        print(f"Ana hata: {e}")
        raise
