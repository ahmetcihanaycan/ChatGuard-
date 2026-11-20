# Sunuma hazırlık

Bu dosya senin çalışma notun; teslim sunumu İngilizcedir.

## Proje ne yapıyor?

Bir mesajı clean/offensive olarak sınıflandırıyor. Offensive çıkan mesajın tamamını sohbet
penceresinde gizliyor. Modelin karar puanını ve CPU üzerinde karar süresini gösteriyor.
Varsayılan model gerçekten INT8 BERT; sonuçlar hazır bir listeye bakılarak üretilmiyor.

## Dört model arasındaki fark

1. TF-IDF + Logistic Regression: Eğitimde sık görülen kelime/ikili kelime örüntülerini sayısal
   özelliklere çevirir. Belirli mesajda önemli ama bütün veri içinde daha az yaygın kelimelere
   daha yüksek ağırlık verir. Logistic regression bu özelliklerden karar öğrenir.
2. Word2Vec + sinir ağı: Word2Vec yalnızca eğitim mesajlarından kelime vektörleri öğrenir.
   Mesajdaki bilinen kelimelerin vektör ortalaması, ReLU ve dropout içeren küçük ağa gider.
   Ağ SGD, momentum ve backpropagation ile eğitilir. Kelime sırasını kaybetmesi bir sınırlamadır.
3. BERT FP32: Önceden genel dil öğrenmiş küçük bir Transformer, bizim eğitim mesajlarımızla
   fine-tune edilir. Self-attention, sözcükleri diğer sözcüklerle ilişkileri içinde işler.
4. BERT INT8: Aynı fine-tune edilmiş modelin Linear katmanları dinamik quantization ile
   düşük hassasiyette çalıştırılır. Embedding ve diğer bazı işlemler FP32 kalır.

## Hoca sorarsa

**Neden küçük İngilizce BERT?** Görev İngilizce; çok dilli modelin maliyetine ihtiyaç yok.
İki katmanlı küçük BERT, CPU'da 50 ms ortalama gecikme hedefi için tercih edildi.

**Bütün ağı sıfırdan mı eğittin?** Word2Vec ve küçük feedforward ağ sıfırdan eğitildi.
BERT'in başlangıç ağırlıkları Google'ın ön eğitiminden geliyor; sınıflandırma için burada
fine-tune edildi. Bu iki öğrenme süreci farklıdır.

**Quantization nedir?** Bazı ağırlıkları 32-bit kayan nokta yerine 8-bit tamsayıyla temsil
etmek. Dinamik yöntemde aktivasyon ölçekleri çalışma sırasında belirlenir. Boyut/hız faydası
olabilir; kaliteyi aynı test setinde ayrıca ölçmek gerekir. Bütün dosyanın dörtte bire
inmesini beklemiyoruz, çünkü embedding tablosu FP32 kalıyor.

**Neden accuracy yerine macro-F1?** Sınıflar dengesiz. Her sınıfın precision ve recall'ından
F1 hesaplanıp sınıflar eşit ağırlıkla ortalanır. Çoğunluk sınıfına sürekli aynı cevabı vermek
accuracy'yi yanıltıcı şekilde artırabilir.

**Veri sızıntısını nasıl önledin?** TF-IDF ve Word2Vec yalnızca train üzerinden öğrenir.
Epoch seçimi validation skoruna dayanır. Test etiketleri model seçiminde kullanılmaz.
Normalize edilmiş train tekrarları ve validation/test ile çakışan train metinleri dışlanır.

**Hız nasıl ölçüldü?** Aynı CPU, iki PyTorch thread'i ve batch 1. 20 ısınma sonrasında 200
mesajın her biri zamanlanır. Ön işleme ve model dahil; tarayıcı/ağ hariç. Ortalama ve p95 ayrı.

**Confidence gerçek doğruluk olasılığı mı?** Hayır. Softmax/logistic model puanı. Kalibrasyon
yapılmadığından yüzde 90 puanlı her 10 karardan 9'unun doğru olduğunu iddia edemeyiz.

**Neden bazı kararlar yanlış?** Eğitim verisi sosyal medya metni; oyun sohbeti farklı.
İroni, alıntı, bağlam eksikliği ve uzun mesajın kesilmesi hata yaratabilir.

**Transformer baseline'dan kötü çıkarsa?** Gerçek ve geçerli bir bulgu. Küçük model kapasitesi,
eğitim verisinin boyutu ve güçlü kelime ipuçları bunu açıklayabilir. Sonuç değiştirilmez.

## Sunmadan önce

Uygulamayı aç, yeni mesajlar dene, dört modelin farkını gözle. `results/metrics.json` içindeki
gerçek hatalardan birini incele. README kurulumunu ve her kod dosyasının görevini açıklayabil.
AI Usage slaytını koru; projede ChatGPT/Codex desteği kullanıldığını açıkça belirt.
