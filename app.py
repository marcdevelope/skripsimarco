# =========================
# IMPORT LIBRARY
# =========================

import streamlit as st
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="ABSA E-Commerce Review Analysis",
    layout="wide"
)

st.title("ABSA E-Commerce Review Analysis")
st.write("Prediksi aspek, sentimen, dan confidence menggunakan model SVM dan IndoBERT.")


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Pengaturan Model")

app_choice = st.sidebar.selectbox(
    "Pilih aplikasi:",
    ["Shopee", "Tokopedia"]
)

model_choice = st.sidebar.selectbox(
    "Pilih model:",
    ["SVM", "IndoBERT"]
)


# =========================
# PATH MODEL
# =========================

if app_choice == "Shopee" and model_choice == "SVM":
    model_path = "model_streamlit_svm_shopee/svm_sentiment_shopee.pkl"

elif app_choice == "Tokopedia" and model_choice == "SVM":
    model_path = "svm_tokopedia.pkl"

elif app_choice == "Shopee" and model_choice == "IndoBERT":
    model_path = "model_streamlit/indobert_shopee.pth"

else:
    model_path = "indobert_tokopedia.pth"


st.sidebar.write("Model path:")
st.sidebar.code(model_path)

if not os.path.exists(model_path):
    st.error(f"File model tidak ditemukan: {model_path}")
    st.stop()


# =========================
# ASPECT KEYWORDS
# =========================

aspect_keywords = {
    "harga": [
        "harga", "mahal", "murah", "diskon", "promo", "ongkir",
        "biaya", "voucher", "cashback", "gratis ongkir", "potongan"
    ],

    "produk": [
        "barang", "produk", "kualitas", "rusak", "original",
        "palsu", "sesuai", "cacat", "bagus", "jelek", "stok"
    ],

    "pengiriman": [
        "kirim", "pengiriman", "kurir", "paket", "sampai",
        "lama", "cepat", "terlambat", "resi", "antar"
    ],

    "sistem aplikasi": [
        "aplikasi", "app", "login", "error", "lemot",
        "bug", "loading", "checkout", "server", "crash",
        "transaksi", "bayar", "payment", "fitur", "voucher tidak bisa"
    ]
}


def predict_aspect(text):
    text = str(text).lower()
    scores = {}

    for aspect, keywords in aspect_keywords.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[aspect] = score

    predicted_aspect = max(scores, key=scores.get)

    # Supaya tidak ada "tidak terdeteksi", fallback ke sistem aplikasi
    if scores[predicted_aspect] == 0:
        predicted_aspect = "sistem aplikasi"
        confidence = 0.50
        return predicted_aspect, confidence, scores

    total_score = sum(scores.values())
    confidence = scores[predicted_aspect] / total_score

    return predicted_aspect, confidence, scores


# =========================
# LOAD SVM MODEL
# =========================

@st.cache_resource
def load_svm_model(path):
    return joblib.load(path)


def normalize_sentiment_label(label):
    label = str(label).lower().strip()

    if label in ["0", "negative", "negative_1", "negative_2", "very_negative"]:
        return "negative"

    elif label in ["1", "neutral"]:
        return "neutral"

    elif label in ["2", "positive"]:
        return "positive"

    else:
        return label


def predict_svm_with_confidence(text, model):
    prediction = model.predict([text])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]

        confidence_per_class = {
            normalize_sentiment_label(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }

        confidence = float(np.max(probabilities))

    else:
        scores = model.decision_function([text])

        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        scores = scores[0]

        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

        confidence_per_class = {
            normalize_sentiment_label(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }

        confidence = float(np.max(probabilities))

    predicted_sentiment = normalize_sentiment_label(prediction)

    return predicted_sentiment, confidence, confidence_per_class


# =========================
# LOAD INDOBERT MODEL
# =========================

@st.cache_resource
def load_indobert_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_path = "model_streamlit/indobert_tokenizer"

    if not os.path.exists(tokenizer_path):
        st.error(f"Folder tokenizer tidak ditemukan: {tokenizer_path}")
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    state_dict = torch.load(path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("model."):
            new_key = new_key.replace("model.", "", 1)

        if new_key.startswith("bert_model."):
            new_key = new_key.replace("bert_model.", "bert.", 1)

        if new_key.startswith("indobert."):
            new_key = new_key.replace("indobert.", "bert.", 1)

        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)

        new_state_dict[new_key] = value

    if "classifier.weight" not in new_state_dict:
        st.error("classifier.weight tidak ditemukan di file .pth.")
        st.stop()

    num_labels = new_state_dict["classifier.weight"].shape[0]

    if num_labels == 3:
        id2label = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

    elif num_labels == 5:
        id2label = {
            0: "negative_1",
            1: "negative_2",
            2: "neutral",
            3: "positive",
            4: "very_negative"
        }

    else:
        st.error(f"Jumlah label model tidak dikenali: {num_labels}")
        st.stop()

    model = AutoModelForSequenceClassification.from_pretrained(
        "indobenchmark/indobert-base-p1",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    try:
        model.load_state_dict(new_state_dict, strict=True)

    except RuntimeError:
        st.warning("Model .pth tidak cocok 100% dengan struktur AutoModelForSequenceClassification.")
        st.info("Mencoba load dengan strict=False.")

        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict,
            strict=False
        )

        if len(missing_keys) > 0:
            st.write("Contoh missing keys:")
            st.write(missing_keys[:10])

        if len(unexpected_keys) > 0:
            st.write("Contoh unexpected keys:")
            st.write(unexpected_keys[:10])

    model.to(device)
    model.eval()

    return model, tokenizer, id2label, device


def predict_indobert_with_confidence(text, model, tokenizer, id2label, device):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_id = predicted_class.item()
    predicted_sentiment = normalize_sentiment_label(id2label[predicted_id])
    sentiment_confidence = confidence.item()

    confidence_per_class = {
        normalize_sentiment_label(id2label[i]): float(probabilities[0][i])
        for i in range(probabilities.shape[1])
    }

    return predicted_sentiment, sentiment_confidence, confidence_per_class


# =========================
# PIE CHART FUNCTION
# =========================

def show_small_pie_chart(df_chart, label_col, value_col, title):
    st.markdown(f"### {title}")

    df_chart = df_chart.copy()
    df_chart = df_chart[df_chart[value_col] > 0]

    if df_chart.empty:
        st.warning("Data tidak cukup untuk membuat pie chart.")
        return

    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=120)

    wedges, texts, autotexts = ax.pie(
        df_chart[value_col],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        textprops={
            "fontsize": 7
        }
    )

    ax.legend(
        wedges,
        df_chart[label_col],
        title=label_col,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        title_fontsize=8
    )

    ax.axis("equal")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# =========================
# GENERATE KESIMPULAN
# =========================

def generate_conclusion(
    app_choice,
    model_choice,
    user_input,
    predicted_aspect,
    aspect_confidence,
    predicted_sentiment,
    sentiment_confidence
):
    sentiment_lower = str(predicted_sentiment).lower()

    if "positive" in sentiment_lower:
        sentiment_meaning = "menunjukkan bahwa pengguna cenderung merasa puas atau memberikan penilaian positif terhadap pengalaman penggunaan aplikasi."
        sentiment_action = "Hal ini dapat menjadi indikator bahwa aspek tersebut sudah memberikan pengalaman yang baik bagi pengguna."

    elif "neutral" in sentiment_lower:
        sentiment_meaning = "menunjukkan bahwa ulasan pengguna bersifat netral atau tidak terlalu menunjukkan kepuasan maupun ketidakpuasan secara kuat."
        sentiment_action = "Hal ini dapat menjadi masukan bahwa aspek tersebut masih perlu dipantau agar tidak berkembang menjadi pengalaman negatif."

    elif "very_negative" in sentiment_lower:
        sentiment_meaning = "menunjukkan bahwa pengguna memiliki tingkat ketidakpuasan yang sangat tinggi terhadap pengalaman penggunaan aplikasi."
        sentiment_action = "Hal ini perlu menjadi perhatian utama karena dapat berdampak pada persepsi negatif pengguna terhadap aplikasi."

    elif "negative" in sentiment_lower:
        sentiment_meaning = "menunjukkan bahwa pengguna cenderung merasa kurang puas atau mengalami kendala pada aplikasi."
        sentiment_action = "Hal ini dapat menjadi masukan bagi pengembang atau pihak terkait untuk melakukan perbaikan pada aspek yang terdeteksi."

    else:
        sentiment_meaning = "menunjukkan adanya kecenderungan sentimen tertentu berdasarkan hasil prediksi model."
        sentiment_action = "Hasil ini dapat digunakan sebagai bahan analisis lanjutan."

    if predicted_aspect == "harga":
        aspect_meaning = "harga, promo, diskon, ongkos kirim, voucher, cashback, atau biaya transaksi"

    elif predicted_aspect == "produk":
        aspect_meaning = "kualitas barang, kondisi produk, kesesuaian barang, atau pengalaman pengguna terhadap produk"

    elif predicted_aspect == "pengiriman":
        aspect_meaning = "proses pengiriman, kurir, estimasi waktu sampai, resi, atau keterlambatan paket"

    elif predicted_aspect == "sistem aplikasi":
        aspect_meaning = "performa aplikasi, error, login, loading, checkout, transaksi, server, atau sistem pembayaran"

    else:
        aspect_meaning = "aspek yang belum dapat dikategorikan secara jelas berdasarkan kata kunci yang tersedia"

    conclusion = f"""
    Berdasarkan hasil analisis menggunakan model **{model_choice}** pada aplikasi **{app_choice}**, 
    ulasan pengguna terdeteksi paling berkaitan dengan aspek **{predicted_aspect}**. 
    Aspek ini berhubungan dengan **{aspect_meaning}**.

    Model memprediksi sentimen ulasan sebagai **{predicted_sentiment}** dengan nilai confidence sebesar 
    **{sentiment_confidence:.4f}**. Sementara itu, nilai confidence untuk aspek adalah **{aspect_confidence:.4f}**. 
    Hasil ini {sentiment_meaning}

    Dengan demikian, kesimpulan dari ulasan ini adalah bahwa pengguna memberikan tanggapan yang berfokus pada aspek 
    **{predicted_aspect}** dengan kecenderungan sentimen **{predicted_sentiment}**. {sentiment_action}
    """

    return conclusion


# =========================
# LOAD MODEL BERDASARKAN PILIHAN
# =========================

try:
    if model_choice == "SVM":
        model = load_svm_model(model_path)
        st.success("Model SVM berhasil diload.")

    else:
        model, tokenizer, id2label, device = load_indobert_model(model_path)
        st.success("Model IndoBERT berhasil diload.")

except Exception as e:
    st.error("Model gagal diload.")
    st.exception(e)
    st.stop()


# =========================
# INPUT USER
# =========================

st.subheader("Input Review Pengguna")

user_input = st.text_area(
    "Masukkan ulasan pengguna:",
    placeholder="Contoh: barang bagus tapi pengiriman sangat lama dan aplikasi sering error"
)


# =========================
# PREDICTION
# =========================

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Masukkan teks ulasan terlebih dahulu.")

    else:
        predicted_aspect, aspect_confidence, aspect_scores = predict_aspect(user_input)

        try:
            if model_choice == "SVM":
                predicted_sentiment, sentiment_confidence, sentiment_probs = predict_svm_with_confidence(
                    user_input,
                    model
                )

            else:
                predicted_sentiment, sentiment_confidence, sentiment_probs = predict_indobert_with_confidence(
                    user_input,
                    model,
                    tokenizer,
                    id2label,
                    device
                )

            st.subheader("Hasil Prediksi")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### Aplikasi")
                st.info(app_choice)

            with col2:
                st.markdown("### Model")
                st.info(model_choice)

            with col3:
                st.markdown("### Input Review")
                st.write(user_input)

            col4, col5 = st.columns(2)

            with col4:
                st.markdown("### Prediksi Aspek")
                st.success(predicted_aspect)
                st.write("Aspect Confidence:", round(aspect_confidence, 4))

            with col5:
                st.markdown("### Prediksi Sentimen")
                st.success(predicted_sentiment)
                st.write("Sentiment Confidence:", round(sentiment_confidence, 4))

            # =========================
            # CONFIDENCE SENTIMENT
            # =========================

            st.subheader("Confidence per Sentiment")

            sentiment_df = pd.DataFrame(
                sentiment_probs.items(),
                columns=["Sentiment", "Confidence"]
            ).sort_values(by="Confidence", ascending=False)

            sentiment_df["Percentage"] = sentiment_df["Confidence"] * 100

            st.dataframe(sentiment_df, use_container_width=True)

            col_pie_sentiment, col_bar_sentiment = st.columns([1, 2])

            with col_pie_sentiment:
                show_small_pie_chart(
                    sentiment_df,
                    label_col="Sentiment",
                    value_col="Percentage",
                    title="Pie Chart Sentimen"
                )

            with col_bar_sentiment:
                st.markdown("### Bar Chart Sentimen")
                bar_sentiment_df = sentiment_df[["Sentiment", "Percentage"]].set_index("Sentiment")
                st.bar_chart(bar_sentiment_df)


            # =========================
            # ASPECT SCORES
            # =========================

            st.subheader("Aspect Scores")

            aspect_df = pd.DataFrame(
                aspect_scores.items(),
                columns=["Aspect", "Score"]
            )

            st.dataframe(aspect_df, use_container_width=True)

            col_pie_aspect, col_bar_aspect = st.columns([1, 2])

            if aspect_df["Score"].sum() > 0:
                aspect_df["Percentage"] = aspect_df["Score"] / aspect_df["Score"].sum() * 100

                with col_pie_aspect:
                    show_small_pie_chart(
                        aspect_df,
                        label_col="Aspect",
                        value_col="Percentage",
                        title="Pie Chart Aspek"
                    )

                with col_bar_aspect:
                    st.markdown("### Bar Chart Aspek")
                    bar_aspect_df = aspect_df[["Aspect", "Score"]].set_index("Aspect")
                    st.bar_chart(bar_aspect_df)

            else:
                st.warning("Tidak ada keyword aspek yang terdeteksi. Aspek otomatis diarahkan ke sistem aplikasi.")

            # =========================
            # KESIMPULAN
            # =========================

            st.subheader("Kesimpulan Hasil Prediksi")

            conclusion_text = generate_conclusion(
                app_choice=app_choice,
                model_choice=model_choice,
                user_input=user_input,
                predicted_aspect=predicted_aspect,
                aspect_confidence=aspect_confidence,
                predicted_sentiment=predicted_sentiment,
                sentiment_confidence=sentiment_confidence
            )

            st.info(conclusion_text)

        except Exception as e:
            st.error("Prediksi gagal.")
            st.exception(e)