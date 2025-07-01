# Import dan Setup
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np #@Albert buang aja, gak kepake
from openai import OpenAI
import time

# ----------------- Konfigurasi Model -----------------
MODEL_CONFIG = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": {
            "DeepSeek-Chat (128K Context)": "deepseek-chat",
            "DeepSeek-Reasoner": "deepseek-reasoner"
        }
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "models": {
            "GPT-3.5 Turbo": "gpt-3.5-turbo",
            "GPT-4 Turbo": "gpt-4-turbo"
        }
    }
}

# ----------------- Backend RAG & LLM -----------------

## Inisiasi Model Sentence Transformer
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

embedding_model = load_embedding_model()

## FAISS & Euclidean Distance
def build_faiss_index(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

## Retrieval
def retrieve(query, index, df, top_k):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k = None)
    return df.iloc[indices[0]]

## LLM - Generate Answer (Enhanced for DeepSeek & OpenAI)
def generate_answer(query, context, api_key, provider, model_name):
    config = MODEL_CONFIG[provider]
    
    client = OpenAI(
        api_key=api_key,
        base_url=config["base_url"]
    )
    
    # System messages khusus untuk model DeepSeek
    if provider == "DeepSeek":
        if "Reasoner" in model_name:
            system_message = "Anda adalah DeepSeek-Reasoner, model AI khusus untuk penalaran logis dan analisis kritis. Jawablah dengan struktur logis dan langkah-langkah penalaran yang jelas."
        else:
            system_message = "Anda adalah DeepSeek-Chat, asisten AI dengan konteks panjang (128K token). Berikan jawaban komprehensif berdasarkan konteks yang diberikan."
    else:
        system_message = "Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan data yang diberikan."
    
    user_message = f"""
    ### Pertanyaan:
    {query}
    
    ### Konteks:
    {context}
    
    ### Instruksi:
    1. Jawab pertanyaan secara langsung dan lugas
    2. Jika informasi tidak cukup, jelaskan keterbatasannya
    3. Gunakan format markdown untuk struktur yang jelas
    4. Untuk pertanyaan analitis, jelaskan langkah-langkah penalaran
    """
    
    # Parameter khusus untuk DeepSeek-Reasoner
    temperature = 0.3 if "Reasoner" in model_name else 0.4
    max_tokens = 4096 if "Reasoner" in model_name else 2048
    
    response = client.chat.completions.create(
        model=config["models"][model_name],
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    return response.choices[0].message.content.strip()

#def transform_data(df, selected_columns):
    #df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
    #return df
def transform_data(df, selected_columns):
    # Reset indeks untuk menghindari konflik
    df_reset = df.reset_index(drop=True)
    
    # Gabungkan kolom dan handle NaN
    df_reset["text"] = (
        df_reset[selected_columns]
        .fillna("N/A")  # Ganti NaN dengan placeholder
        .astype(str)
        .agg(" | ".join, axis=1)
    )
    return df_reset

# ----------------- UI -----------------
st.set_page_config(
    page_title="DeepSeek RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header Modern
st.markdown("""
<div style="
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
">
    <h1 style='text-align: center; margin: 0;'>DeepSeek RAG Assistant</h1>
    <p style='text-align: center; font-size: 1.2rem;'>
        Analisis Data dengan Model DeepSeek-Chat & DeepSeek-Reasoner
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <h2>⚙️ Konfigurasi Model</h2>
        <p>Pilih provider dan model AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Provider Selection
    provider = st.radio(
        "Pilih Provider API:",
        ["DeepSeek", "OpenAI"],
        index=0,
        horizontal=True
    )
    
    # Model Selection based on provider
    model_name = st.selectbox(
        f"Pilih Model {provider}:",
        list(MODEL_CONFIG[provider]["models"].keys())
    )
    
    # API Key Input
    api_key = st.text_input(f"🔑 {provider} API Key", type='password')
    activate_btn = st.button('Aktifkan API Key', use_container_width=True)
    
    if activate_btn and api_key:
        st.session_state.api_key = api_key
        st.session_state.provider = provider
        st.session_state.model_name = model_name
        st.success(f"{provider} API Diaktifkan!")
    
    st.divider()
    uploaded_file = st.file_uploader("📤 Upload File CSV", type='csv')
    
    # Model information
    st.divider()
    st.subheader("ℹ️ Info Model")
    if provider == "DeepSeek":
        if "Reasoner" in model_name:
            st.info("""
            **DeepSeek-Reasoner**:
            - Khusus untuk penalaran logis dan analisis kritis
            - Optimasi untuk tugas berpikir langkah-demi-langkah
            - Token maks: 4,096
            """)
        else:
            st.info("""
            **DeepSeek-Chat**:
            - Konteks panjang (128K token)
            - Cocok untuk dokumen besar
            - Kemampuan pemahaman umum
            """)
    else:
        st.info(f"""
        **{model_name}**:
        - Model state-of-the-art dari OpenAI
        - Cocok untuk berbagai tugas umum
        """)

# Main Content
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file,header=0)
        
        st.subheader("🗂 Preview Data")
        st.write(f"File: {uploaded_file.name}")
        
        # Column selection
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect(
            "Pilih kolom untuk analisis:",
            options=all_columns,
            default=all_columns[:min(5, len(all_columns))]
        )
        
        if not selected_columns:
            st.warning("Silakan pilih minimal satu kolom.")
            st.stop()

        # @Albert penyakitnya ini disini, kalau head 5 berarti kasih 5 row aja dari DataFramenya
        # Lo bisa ganti jadi kek gini kalau mau semuanya muncul >> st.dataframe(df[selected_columns])
        # Atau gak bisa juga tambahin interaksi kek gini aja pake slidernya streamlit
        # num_rows = st.slider("Berapa baris yang mau ditampilkan?", 5, min(100, len(df)), 5)
        # st.dataframe(df[selected_columns].head(num_rows)) 
        st.write(f"Total baris: {len(df)}")    
        st.dataframe(df[selected_columns]) 
        df = transform_data(df, selected_columns)
        
        # Query input
        st.subheader("💬 Ajukan Pertanyaan")
        query = st.text_area("Masukkan pertanyaan tentang data Anda:", height=120)
        run_query = st.button("🚀 Dapatkan Jawaban", use_container_width=True)
        
        if run_query:
            if 'api_key' not in st.session_state:
                st.warning("Silakan aktifkan API key terlebih dahulu")
                st.stop()
                
            # Show current configuration
            st.markdown(f"""
            <div style="
                background: #f0f2f6;
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
            ">
                <b>Konfigurasi Saat Ini:</b><br>
                Provider: <code>{st.session_state.provider}</code> | 
                Model: <code>{st.session_state.model_name}</code>
            </div>
            """, unsafe_allow_html=True)
            
            # Processing pipeline
            with st.status("🔍 Memproses permintaan Anda...", expanded=True) as status:
                st.write("Membangun indeks pencarian...")
                index, _ = build_faiss_index(df['text'].to_list())
                
                st.write("Mencari data relevan...")
                results = retrieve(query, index, df)
                context = "\n".join([
                    f"📌 Dokumen {i+1}:\n{row['text']}\n" 
                    for i, (_, row) in enumerate(results.iterrows())
                ])
                
                st.write(f"Menghasilkan jawaban menggunakan {st.session_state.model_name}...")
                start_time = time.time()
                answer = generate_answer(
                    query, 
                    context, 
                    st.session_state.api_key,
                    st.session_state.provider,
                    st.session_state.model_name
                )
                processing_time = time.time() - start_time
                
                status.update(label="✅ Pemrosesan selesai!", 
                              state="complete", 
                              expanded=False)
            
            # Display results
            st.subheader("💡 Jawaban")
            st.markdown(f"**Waktu Pemrosesan:** {processing_time:.2f} detik")
            st.divider()
            st.markdown(answer)
            
            # Show context
            with st.expander("🔍 Lihat data relevan yang digunakan"):
                st.dataframe(results[selected_columns])
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
else:
    # Welcome screen
    st.info("Silakan upload file CSV untuk memulai analisis")
    
    # Container untuk welcome content
    with st.container():
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            # Card Cara Penggunaan
            with st.container():
                st.markdown("""
                <div style="
                    background: white;
                    border-radius: 10px;
                    padding: 1.5rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    margin-bottom: 1rem;
                ">
                    <h3 style="color: #2563eb;">✨ Cara Penggunaan</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                1. Pilih provider API (DeepSeek/OpenAI)
                2. Pilih model yang sesuai
                3. Masukkan API Key
                4. Upload file CSV
                5. Pilih kolom untuk analisis
                6. Ajukan pertanyaan tentang data Anda
                """)
            
            # Card Mendapatkan API Key
            with st.container():
                st.markdown("""
                <div style="
                    background: white;
                    border-radius: 10px;
                    padding: 1.5rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h3 style="color: #2563eb;">🔑 Mendapatkan API Key</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                - **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com)
                - **OpenAI**: [platform.openai.com](https://platform.openai.com)
                """)
                st.caption("API Key disimpan secara lokal di browser Anda dan tidak dikirim ke server manapun")
        
        with col2:
            # Container untuk Model DeepSeek
            with st.container():
                st.markdown("<h3 style='color: #2563eb; text-align: center;'>🚀 Model DeepSeek</h3>", 
                           unsafe_allow_html=True)
                
                # Card DeepSeek-Chat
                with st.container():
                    st.markdown(
                        f"<div style='background: linear-gradient(135deg, #e0f7fa 0%, #bbdefb 100%); "
                        f"border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;'>"
                        f"<h4 style='color: #0288d1; margin: 0;'>DeepSeek-Chat</h4>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                    st.markdown("""
                    - Konteks 128K token
                    - Pemahaman dokumen panjang
                    - Cocok untuk analisis data umum
                    """)
                
                # Card DeepSeek-Reasoner
                with st.container():
                    st.markdown(
                        f"<div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); "
                        f"border-radius: 8px; padding: 1.2rem;'>"
                        f"<h4 style='color: #388e3c; margin: 0;'>DeepSeek-Reasoner</h4>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                    st.markdown("""
                    - Spesialis penalaran logis
                    - Analisis langkah-demi-langkah
                    - Optimasi untuk tugas kritis
                    """)
                
                # Tombol Kunjungi DeepSeek
                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.link_button("Kunjungi DeepSeek", "https://deepseek.com", use_container_width=True)
    
    # Spasi vertikal
    st.write("")
    st.write("")
    
    # Contoh penggunaan
    with st.expander("📚 Contoh Pertanyaan Analitik"):
        st.markdown("""
        Berikut beberapa contoh pertanyaan yang bisa Anda ajukan setelah mengupload data:
        
        - **Analisis Tren**: "Apa tren penjualan bulanan untuk produk A selama tahun 2023?"
        - **Perbandingan**: "Bandingkan performa penjualan antara wilayah timur dan barat"
        - **Prediksi**: "Berdasarkan data historis, prediksi penjualan untuk kuartal berikutnya"
        - **Korelasi**: "Apakah ada korelasi antara pengeluaran pemasaran dan peningkatan penjualan?"
        - **Segmentasi**: "Identifikasi segmen pelanggan dengan nilai lifetime tertinggi"
        """)

# Footer (selalu muncul di bawah)
st.divider()
st.caption("DeepSeek RAG Assistant • Dibangun dengan Streamlit • Support DeepSeek-Chat & DeepSeek-Reasoner • v1.0")

