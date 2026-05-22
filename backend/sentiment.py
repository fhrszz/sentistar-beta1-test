from transformers import pipeline
from functools import lru_cache

# ── Kata slang yang menurunkan kepercayaan model ───────────────────────────
SLANG_KEYWORDS = {
    "skibidi", "sigma", "rizz", "gyatt", "bussin", "slay", "based",
    "anjay", "anjir", "wkwk", "btw", "otw", "bgt", "kuy", "gws",
    "mager", "baper", "gabut", "mantul", "kepo", "php", "bucin",
}

LABEL_MAP = {
    "positive": "Positif",
    "neutral":  "Netral",
    "negative": "Negatif",
}


@lru_cache(maxsize=1)
def get_classifier():
    """
    Load model satu kali saat pertama kali dipanggil, lalu cache.
    Ini mencegah model di-reload setiap request.
    """
    print("⏳ Loading model sentiment...")
    clf = pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier"
    )
    print("✅ Model loaded")
    return clf


def rating_to_scores(bintang: int) -> dict:
    """Konversi rating 1–5 ke distribusi probabilitas tiga label."""
    mapping = {
        1: {"positive": 0.05, "neutral": 0.10, "negative": 0.85},
        2: {"positive": 0.10, "neutral": 0.20, "negative": 0.70},
        3: {"positive": 0.20, "neutral": 0.60, "negative": 0.20},
        4: {"positive": 0.70, "neutral": 0.20, "negative": 0.10},
        5: {"positive": 0.85, "neutral": 0.10, "negative": 0.05},
    }
    return mapping[bintang]


def hitung_bobot_dinamis(
    teks: str,
    skor_model: float,
    bobot_model_default: float = 0.6
) -> float:
    """
    Turunkan bobot model secara dinamis jika:
    - Teks mengandung slang         → penalti 0.25
    - Teks sangat pendek (≤4 kata)  → penalti 0.10
    - Confidence model rendah (<70%) → penalti 0.15
    """
    kata = set(teks.lower().split())
    ada_slang   = bool(kata & SLANG_KEYWORDS)
    teks_pendek = len(kata) <= 4
    conf_rendah = skor_model < 0.70

    penalti = 0.0
    if ada_slang:   penalti += 0.25
    if teks_pendek: penalti += 0.10
    if conf_rendah: penalti += 0.15

    return round(max(0.15, bobot_model_default - penalti), 2)


def analisis_sentimen(
    teks: str,
    bintang: int,
    bobot_model_default: float = 0.6
) -> dict:
    """
    Gabungkan prediksi model NLP dengan sinyal rating bintang.

    Returns dict dengan key:
        label_model, conf_model, bobot_model, bobot_rating,
        label_akhir, skor_akhir, ada_slang, berubah
    """
    classifier = get_classifier()

    # ── Prediksi model ──────────────────────────────────────────────
    hasil_model = classifier(teks)[0]
    label_model = hasil_model["label"].lower()   # positive / neutral / negative
    skor_model  = hasil_model["score"]

    # ── Bobot dinamis ───────────────────────────────────────────────
    bobot_model  = hitung_bobot_dinamis(teks, skor_model, bobot_model_default)
    bobot_rating = round(1.0 - bobot_model, 2)

    # ── Distribusi penuh dari model ─────────────────────────────────
    distribusi_model = {
        lb: (1 - skor_model) / 2
        for lb in ["positive", "neutral", "negative"]
    }
    distribusi_model[label_model] = skor_model

    # ── Distribusi dari rating ──────────────────────────────────────
    distribusi_rating = rating_to_scores(bintang)

    # ── Weighted fusion ─────────────────────────────────────────────
    fusi = {
        lb: bobot_model * distribusi_model[lb] + bobot_rating * distribusi_rating[lb]
        for lb in ["positive", "neutral", "negative"]
    }

    label_akhir = max(fusi, key=fusi.get)
    skor_akhir  = fusi[label_akhir]

    kata      = set(teks.lower().split())
    ada_slang = bool(kata & SLANG_KEYWORDS)

    return {
        "label_model" : LABEL_MAP[label_model],
        "conf_model"  : round(skor_model, 4),
        "bobot_model" : bobot_model,
        "bobot_rating": bobot_rating,
        "label_akhir" : LABEL_MAP[label_akhir],
        "skor_akhir"  : round(skor_akhir, 4),
        "ada_slang"   : ada_slang,
        "berubah"     : label_model != label_akhir,
    }
