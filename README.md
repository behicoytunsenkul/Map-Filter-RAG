# Map-Filter-RAG
Klasik RAG yapılarında, Türkçe dilinde bariz olan birçok sorun bulunmaktadır:
* Özellikle Türkçe gibi eklemeli (agglutinative) dillerde doğru arama yapmak zorlaşıyor çünkü kelimelerin yüzlerce farklı çekim hali var.
* Çok fazla belge getirilirse model bunların hepsini işleyemiyor.
* Retriever yanlış veya alakasız içerik getirirse, cevap da yanlış oluyor.
  
MAP-FILTER-RAG, klasik RAG'e eklenen iki aşamalı çalışan doğrulama/temizleme katmanlarına sahip yeni bir RAG metodolojisidir:
* MAP: Getirilen belgeler veya pasajlar tek tek değerlendirilir, her biri soruyla ne kadar alakalı olduğuna göre işaretlenir.
Yani “bu doküman soruya yanıt veriyor mu?” diye ayrı ayrı bakılır.
* FILTER: Eşleme aşamasında alakasız bulunan veya düşük güvenliğe sahip içerikler ayıklanır.
Böylece LLM’e sadece yüksek kaliteli, soruyla alakalı belgeler gönderilir.

**Sonuç:** Modelin önüne çöp bilgi gitmez → daha doğru ve odaklı yanıt gelir.

# MAP-FILTER RAG’in Artıları
* Daha az halüsinasyon: Model “konuyla ilgisiz” metinleri kullanmaz.
* Kısa ve odaklı yanıtlar: Bilgi gürültüsü azalır.
* Özellikle uzun belgelerde işe yarar, çünkü modelin bağlam penceresi sınırlıdır.

# Türkçe'de Neden Daha Başarılıdır?
Türkçe, çekim ve eklemelerle çok farklı kelime formları üreten bir dildir:
* Örn: kitap, kitapta, kitaptan, kitaplarımızdan, kitapçılardan…
* Basit bir “kelime eşleştirme” yöntemi çoğu formu yakalayamaz.
Klasik RAG bu yüzden Türkçe’de sıklıkla şunlara takılır:
* Yanlış belgeleri getirme,
* Çok az belge bulma,
* Semantik olarak yakın ama biçimsel olarak farklı kelimeleri atlamaz.

# Genel Olarak
MAP-FILTER RAG, RAG’in “alakasız bilgi getirme” sorununu çözen gelişmiş bir yaklaşım. Türkçe gibi eklemeli dillerde çok daha başarılı çünkü anlamsal kontrol ve filtreleme sayesinde morfolojik çeşitliliği daha iyi yönetiyor. Sonuç olarak: daha güvenilir, doğru ve odaklı cevaplar üretiyor.

