# 🏛 Bodrum Plan Notları RAG Sistemi

**Retrieval-Augmented Generation (RAG)** sistemi — Bodrum ilçesindeki imar planları ve yapılaşma koşulları hakkında Türkçe soru-cevap.

Mimarlar Odası Bodrum Temsilciliği'nin [Plan Notları](https://bodrummimarlarodasi.org.tr/plan-notlari/) sayfasındaki verileri tarar, temizler, vektör veritabanında indeksler ve doğal dil soruları ile arama yapmanıza olanak tanır.

---

## 📋 Özellikler

| Özellik | Açıklama |
|---|---|
| **Web Scraping** | Plan notları sayfasını ve Google Drive bağlantılarını otomatik tarar |
| **Türkçe NLP** | Türkçe karakter normalizasyonu ve metin temizleme |
| **Token-Aware Chunking** | `tiktoken` ile ölçülen 500–800 token'lık parçalara böler |
| **NVIDIA Embeddings** | `nvidia/llama-nemotron-embed-1b-v2` ile yoğun vektör üretimi |
| **FAISS Vector DB** | Kosinüs benzerliği ile hızlı vektör araması |
| **RAG Pipeline** | Bağlam zenginleştirilmiş yanıt üretimi (LLM) |
| **İnteraktif CLI** | Renkli terminal arayüzü ile soru-cevap |

---

## 🏗 Proje Yapısı

```
bodrumrag/
├── .env                  # API anahtarları ve yapılandırma
├── .env.example          # .env şablonu
├── .gitignore
├── pytest.ini            # Test yapılandırması
├── requirements.txt      # Python bağımlılıkları
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py         # Merkezi yapılandırma (dataclass)
│   ├── scraper.py        # Web scraping modülü
│   ├── cleaner.py        # Metin temizleme & Türkçe normaliz.
│   ├── chunker.py        # Token-aware metin parçalama
│   ├── embedder.py       # NVIDIA NIM embedding API istemcisi
│   ├── vector_db.py      # FAISS vektör veritabanı sarmalayıcı
│   ├── retriever.py      # RAG: arama + prompt + LLM yanıt
│   ├── pipeline.py       # İndeksleme pipeline'ı
│   └── cli.py            # Komut satırı arayüzü
├── tests/
│   ├── __init__.py
│   ├── test_cleaner.py
│   ├── test_chunker.py
│   ├── test_vector_db.py
│   ├── test_scraper.py
│   └── test_retriever.py
└── data/                 # (otomatik oluşturulur)
    └── faiss_index.*     # Kaydedilmiş FAISS indeksi
```

---

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.10+
- pip

### 2. Bağımlılıkları Yükleyin

```bash
cd bodrumrag
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. API Anahtarını Ayarlayın

```bash
cp .env.example .env
# .env dosyasını düzenleyerek NVIDIA API anahtarınızı girin
```

**Gerekli anahtarlar:**
- `NVIDIA_API_KEY` — [NVIDIA API Catalog](https://build.nvidia.com/) üzerinden alınır
- `LLM_API_KEY` — Aynı NVIDIA anahtarı kullanılabilir

---

## 💻 Kullanım

### Adım 1: İndeksleme (Scrape + Embed)

Sayfayı tarayıp vektör veritabanını oluşturun:

```bash
python -m src.cli index
```

Bu komut sırasıyla:
1. Plan notları sayfasını tarar
2. Metni temizler ve Türkçe normaliz. yapar
3. Token-aware parçalara böler
4. NVIDIA API ile embedding üretir
5. FAISS indeksini `data/` klasörüne kaydeder

### Adım 2: Soru Sorma

#### İnteraktif Mod (REPL)

```bash
python -m src.cli ask
```

```
╔══════════════════════════════════════════════════════════╗
║          🏛  BODRUM PLAN NOTLARI RAG SİSTEMİ  🏛          ║
║     Bodrum imar planları için akıllı soru-cevap sistemi    ║
╚══════════════════════════════════════════════════════════╝

Soru ▶ Gümüşlük'te yapılaşma koşulları nelerdir?

CEVAP:
Gümüşlük bölgesinde yapılaşma koşulları şu şekildedir...

KAYNAKLAR:
  1. [Benzerlik: 0.923] Gümüşlük_PLAN NOTLARI 001...
  2. [Benzerlik: 0.891] Gümüşlük_PLAN NOTLARI 003...
```

#### Tek Seferlik Soru

```bash
python -m src.cli query "Konacık'ta TAKS ve KAKS değerleri nedir?"
```

### Seçenekler

```bash
# Daha detaylı log çıktısı
python -m src.cli -v ask

# Top-k sonuç sayısını değiştirme
python -m src.cli ask -k 10

# Yardım
python -m src.cli --help
```

---

## 🧪 Testler

```bash
# Tüm testleri çalıştır
pytest

# Belirli bir modülü test et
pytest tests/test_cleaner.py -v

# Kapsam raporu
pip install pytest-cov
pytest --cov=src --cov-report=term-missing
```

---

## ⚙️ Yapılandırma

Tüm ayarlar `.env` dosyasında yapılır:

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `NVIDIA_API_KEY` | — | NVIDIA NIM API anahtarı |
| `NVIDIA_EMBED_MODEL` | `nvidia/llama-nemotron-embed-1b-v2` | Embedding modeli |
| `LLM_MODEL` | `nvidia/llama-3.3-nemotron-super-49b-v1` | Yanıt üretim modeli |
| `CHUNK_SIZE` | `600` | Chunk başına maks token |
| `CHUNK_OVERLAP` | `100` | Chunk'lar arası örtüşme (token) |
| `TOP_K` | `5` | Döndürülecek en benzer chunk sayısı |
| `FAISS_INDEX_PATH` | `data/faiss_index` | İndeks dosya yolu |

---

## 🔧 Mimari

```
┌─────────────┐     ┌──────────┐     ┌──────────┐
│   Scraper   │────▶│ Cleaner  │────▶│ Chunker  │
│ (requests + │     │ (Turkish │     │ (tiktoken │
│    BS4)     │     │   NLP)   │     │  aware)  │
└─────────────┘     └──────────┘     └────┬─────┘
                                          │
                                          ▼
┌─────────────┐     ┌──────────┐     ┌──────────┐
│   CLI /     │◀────│Retriever │◀────│ Embedder │
│   User      │     │  (RAG)   │     │ (NVIDIA  │
│  Interface  │     │          │     │   NIM)   │
└─────────────┘     └────┬─────┘     └────┬─────┘
                         │                │
                         ▼                ▼
                    ┌──────────┐     ┌──────────┐
                    │   LLM    │     │  FAISS   │
                    │ (NVIDIA) │     │ VectorDB │
                    └──────────┘     └──────────┘
```

---

## 📝 Lisans

Bu proje açık kaynaklıdır. Eğitim ve araştırma amaçlı serbestçe kullanılabilir.
