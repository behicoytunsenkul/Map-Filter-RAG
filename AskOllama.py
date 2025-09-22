import json
import time
import re
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import ollama

MILVUS_HOST = "0.0.0.0"
MILVUS_PORT = "19530"
COLLECTION_NAME = "MilvusCollection"
TOP_K_SEARCH = 25

OLLAMA_MODEL_MAP = 'qwen3:1.7b'
OLLAMA_MODEL_REDUCE = 'qwen3:8b'

try:
    embedding_model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    EMBEDDING_DIM = 768
    print("Embedding modeli başarıyla yüklendi.")
except Exception as e:
    print(f"Embedding modeli yüklenirken hata oluştu: {e}")
    exit()

try:
    with open("chunk_texts.json", "r", encoding="utf-8") as f:
        chunk_texts = json.load(f)
        for item in chunk_texts:
            if isinstance(item, dict) and "text" in item:
                item["text"] = item["text"].lower()
    print("Chunk metinleri başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: 'chunk_texts.json' dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
    exit()


def getEmbedding(texts):
    try:
        return embedding_model.encode(texts, convert_to_numpy=True).tolist()
    except Exception as e:
        print(f"Embedding hatası: {e}")
        return None


def searchMilvus(query, collection):
    query_embedding = getEmbedding([query])
    if not query_embedding:
        return None

    searchParams = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=searchParams,
        limit=TOP_K_SEARCH
    )
    return results[0] if results else []


def callAPI(model, msg, temp=0.1):
    try:
        response = ollama.chat(
            model=model,
            messages=msg,
            options={"temperature": temp}
        )
        content = response['message']['content']
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        print(f"Ollama API hatası (Model: {model}): {e}")
        return "API çağrısı sırasında bir hata oluştu."



def main():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        print("Milvus veritabanına başarıyla bağlanıldı ve koleksiyon yüklendi.")
    except Exception as e:
        print(f"Milvus bağlantı hatası: {e}")
        return

    print(f"Filtreleme Modeli: {OLLAMA_MODEL_MAP} | Cevaplama Modeli: {OLLAMA_MODEL_REDUCE}")
    print("(Çıkmak için 'exit' yazınız)")

    while True:
        query = input("\nSoru: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Program sonlandırılıyor.")
            break
        if not query:
            print("Lütfen geçerli bir soru giriniz.")
            continue

        start_time = time.time()

        vectoResults = searchMilvus(query, collection)
        if not vectoResults:
            print("Bu konuda bilgim yok.")
            continue

        retrieved_chunks = [chunk_texts[result.id] for result in vectoResults]

        filtered_original_chunks = []
        print(
            f"\n-- MAP & FILTER Aşaması Başladı: {len(retrieved_chunks)} chunk, '{OLLAMA_MODEL_MAP}' ile analiz ediliyor... ---")

        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get("text", "")
            print(f"Chunk {i + 1}/{len(retrieved_chunks)} işleniyor...", end="")

            map_prompt = (
                f"Aşağıda bir 'METİN PARÇASI' ve bir 'KULLANICI SORUSU' bulunmaktadır. "
                f"Görevin, metin parçasının kullanıcı sorusunu yanıtlamak için doğrudan ilgili olup olmadığına karar vermektir.\n\n"
                f"- Eğer metin parçası soruyla doğrudan ilgiliyse, sadece 'EVET' yaz.\n"
                f"- Eğer metin parçası soruyla ilgili değilse, sadece 'ALAKASIZ' yaz.\n\n"
                f"Başka hiçbir açıklama veya yorum ekleme. Cevabın sadece bu iki kelimeden biri olmalıdır.\n\n"
                f"--METİN PARÇASI---\n{chunk_text}\n\n"
                f"--KULLANICI SORUSU---\n{query}\n\n"
                f"--CEVAP (EVET veya ALAKASIZ)---\n"
            )
            map_msg = [{"role": "user", "content": map_prompt}]
            decision = callAPI(OLLAMA_MODEL_MAP, map_msg, temp=0.0)

            if "EVET" in decision.upper():
                filtered_original_chunks.append(chunk_text)
                print(f" -> Sonuç: ALAKALI (Listeye Eklendi)")
            else:
                print(f" -> Sonuç: ALAKASIZ")

        print(f"\n- MAP & FILTER Aşaması Tamamlandı: {len(filtered_original_chunks)} adet alakalı chunk bulundu. ---")

        finalAnswer = ""
        if not filtered_original_chunks:
            finalAnswer = "Bu konuda dokümanda bilgi bulunmuyor."
        else:
            print(f"--- REDUCE Aşaması Başladı: Nihai cevap '{OLLAMA_MODEL_REDUCE}' ile oluşturuluyor... ---")
            # DEĞİŞİKLİK: Yeni liste, nihai cevabı oluşturmak için birleştiriliyor.
            final_context = "\n\n---\n\n".join(filtered_original_chunks)

            reduce_prompt = (
                "Sen bir sanal asistansın. "
                "Her zaman sadece Türkçe dilinde ve kibar bir şekilde cevap ver. "
                "Cevaplarını yalnızca verilen markdown formatındaki dokümanlardan üret, asla kendi bilginle tamamlamaya çalışma. "
                "Kullanıcı sorusu ile dokümanları tek tek tara ve cevabını bulmaya çalışmalısın"
                "Sadece soruya cevap ver, soru ile ilgisi olmayan cevapları üretme."
                "Eğer yoksa 'Bu konu hakkında bilgim yok' şeklinde bulamadığını söyle."
                f"### DOKÜMAN ###\n{final_context}\n\n"
                f"### KULLANICI SORUSU ###\n{query}\n\n"
            )
            reduce_msg = [{"role": "user", "content": reduce_prompt}]
            finalAnswer = callAPI(OLLAMA_MODEL_REDUCE, reduce_msg, temp=0.1)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\nCevap: {finalAnswer}")
        print(f"Süre: {elapsed_time:.2f} saniye")


if __name__ == "__main__":
    main()
