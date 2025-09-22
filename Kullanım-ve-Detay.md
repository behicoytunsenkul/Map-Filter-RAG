# MAP-FILTER RAG Tabanlı Soru-Cevap Sistemi

Bu proje, Türkçe PDF dokümanlardan bilgi çıkarımı yapmak için geliştirilmiş bir Retrieval-Augmented Generation (RAG) sistemidir.  
Sistem, klasik RAG’in üzerine eklenen MAP-FILTER yaklaşımıyla geliştirilmiştir. Bu sayede:

- Alakasız doküman parçaları filtrelenir.
- LLM’e yalnızca alakalı içerikler gönderilir.
- Türkçe gibi morfolojik açıdan zengin dillerde daha doğru ve güvenilir cevaplar üretilir.

## Teknolojiler

- **Python 3.10+**
- **Milvus** → Vektör veritabanı
- **Sentence-Transformers** → Türkçe embedding modeli
- **Ollama** → LLM çalıştırma (Qwen modelleri)
- **PyPDF2** → PDF okuma
- **NLTK** → Metin işleme


## Vektör Veritabanı Oluşturma (vecdb_builder.py)

MilvusCreate.py dosyası, PDF dokümanlarını okuyup embedding’lere dönüştürerek Milvus’a kaydeder.

### Adımlar:

#### PDF’leri yükleme:
PDFs/ klasöründeki tüm PDF’ler okunur ve sayfa bazlı metin çıkarılır.

#### Temizlik:
Metinler normalize edilir (lowercase, gereksiz boşluklar silinir).

#### Embedding oluşturma:
Türkçe için optimize edilmiş `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` modeli kullanılır.  
Her chunk → 768 boyutlu embedding vektörüne dönüştürülür.

#### Milvus koleksiyonu oluşturma:

- Koleksiyon adı: `MilvusCollection`
- Alanlar:
  - `id` (INT64, primary key)
  - `embedding` (FLOAT_VECTOR, dim=768)
- HNSW indeksi (COSINE benzerliğiyle)

#### Veri ekleme:
Embeddingler batch halinde Milvus’a yazılır.

#### JSON kaydı:
Chunk metinleri `chunk_texts.json` dosyasına kaydedilir.  
Bu dosya daha sonra QA aşamasında kaynak olarak kullanılacaktır.

## MAP-FILTER RAG ile Soru-Cevap (AskOllama.py)

Bu dosya, kullanıcıdan soru alır ve MAP-FILTER RAG yöntemini uygular.

### Çalışma Akışı:

1. **Sorgu embedding’i çıkarılır.**  
   Kullanıcı sorusu → SentenceTransformer embedding.

2. **Milvus araması yapılır.**  
   En alakalı TOP-25 chunk getirilir.

3. **MAP aşaması (Chunk değerlendirme):**  
   Her chunk için küçük model (`qwen3:1.7b`) çalıştırılır.  
   Modelden sadece “EVET” (alakalı) veya “ALAKASIZ” yanıtı istenir.

4. **FILTER aşaması:**  
   - EVET → Chunk tutulur.  
   - ALAKASIZ → Chunk atılır.

5. **REDUCE aşaması (Final cevaplama):**  
   Geriye kalan chunk’lar birleştirilir.  
   Daha büyük model (`qwen3:8b`) kullanılarak nihai cevap üretilir.  
   Cevap sadece dokümanlardan üretilir, model kendi bilgisini eklemez.

## Kullanım

1. Milvus ve Ollama’yı başlatın  
   ```
   milvus start
   ollama serve
   ```

2. PDF’leri embedding’e çevirin  
   ```
   python CreateMilvus.py
   ```

3. Soru-cevap terminalini başlatın  
   ```
   python AskOllama.py
   ```

4. Örnek terminal çıktısı  
   ```
   AI Soru-Cevap Sistemi
   Filtreleme Modeli: qwen3:1.7b | Cevaplama Modeli: qwen3:8b
   (Çıkmak için 'exit' yazınız)

   Soru: BES nedir?

   --- MAP & FILTER Aşaması Başladı: 25 chunk analiz ediliyor ---
   Chunk 1 -> Sonuç: ALAKALI (Listeye Eklendi)
   Chunk 2 -> Sonuç: ALAKASIZ
   ...
   --- MAP & FILTER Aşaması Tamamlandı: 7 adet alakalı chunk bulundu. ---

   --- REDUCE Aşaması Başladı ---
   Cevap: "Uzay teleskopları nasıl çalışır?"
   Süre: 4.23 saniye
   ```

## Öne Çıkanlar

- **MAP-FILTER tekniği:** Alakasız chunk’lar ayıklanarak daha doğru yanıtlar.
- **Türkçe optimizasyonu:** Türkçe için özel eğitilmiş embedding modeli.
- **Hızlı ve modüler:** Milvus + Ollama entegrasyonu.
- **Esnek:** Farklı modellerle kolayca çalışacak şekilde yapılandırılmıştır.

## Notlar

- `OLLAMA_MODEL_MAP`: Küçük model, chunk filtreleme için.
- `OLLAMA_MODEL_REDUCE`: Büyük model, nihai cevaplama için.
- Chunk boyutu ve `TOP_K_SEARCH` değeri, performans ihtiyacına göre ayarlanabilir.
