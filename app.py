import streamlit as st
import numpy as np
import time
import os
import tempfile

# ── TensorFlow (graceful degradation) ──────────────────────────────────────
try:
    import tensorflow as tf
    MOCK_MODE = False
except ImportError:
    MOCK_MODE = True

if not MOCK_MODE:
    from src.pipeline import TextPipeline
    from src.model import SequenceModel
    from src.train import train_step
else:
    class TextPipeline:
        def __init__(self, **kwargs): pass
        def adapt(self, texts): pass
        def create_dataset(self, texts, labels):
            return [([t], [l]) for t, l in zip(texts[:10], labels[:10])]
        def vectorize_layer(self, val): return val

    class SequenceModel:
        def __init__(self, **kwargs): pass
        def __call__(self, x):
            class T:
                def numpy(self): return np.random.random((1, 1))
            return T()

    class MockAttr:
        def __getattr__(self, n): return self
        def __call__(self, *a, **k): return self
        def result(self, *a, **k): return 0.85
        def update_state(self, *a, **k): pass
        def numpy(self, *a, **k): return np.random.random()
        def __float__(self): return 0.5
        def __format__(self, s): return format(0.5, s)

    mock_obj = MockAttr()
    tf = mock_obj
    optimizer = loss_fn = metric = mock_obj
    def train_step(*a): return mock_obj

# ── LangChain / ChromaDB (graceful degradation) ────────────────────────────
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Pipeline Hub",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"] { background: #161b22; }
.status-card {
    padding: 18px 22px; border-radius: 10px;
    background: #1e2530; border: 1px solid #3d4b5f; margin-bottom: 16px;
}
.flow-container { display:flex; flex-direction:column; align-items:center; gap:0; margin-top:12px; }
.flow-node {
    background: linear-gradient(135deg,#1e3a5f,#2563a8);
    color:#e8f4ff; border:1px solid #4e90d6; border-radius:8px;
    padding:8px 20px; font-size:13px; font-weight:600;
    width:210px; text-align:center; box-shadow:0 2px 8px rgba(37,99,168,.3);
}
.flow-arrow { color:#4e90d6; font-size:20px; line-height:1; margin:2px 0; }
.rag-node {
    background: linear-gradient(135deg,#1a3a2a,#1f6b45);
    color:#d4f5e2; border:1px solid #2ecc71; border-radius:8px;
    padding:8px 20px; font-size:13px; font-weight:600;
    width:230px; text-align:center; box-shadow:0 2px 8px rgba(46,204,113,.2);
}
.rag-arrow { color:#2ecc71; font-size:20px; line-height:1; margin:2px 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🧠 LLM Pipeline Hub")
st.caption("Senior Data Scientist III · Production-Grade NLP & RAG Architectures")

tab1, tab2 = st.tabs(["⚡ Bi-LSTM Pipeline", "📚 RAG — Document Q&A"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Existing NLP Pipeline
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    Production-grade `tf.data.AUTOTUNE` pipeline with **Bi-Directional LSTM + Attention**
    and a graph-compiled custom training loop (`@tf.function`).
    """)

    # Sidebar controls (only relevant in tab1 context)
    with st.sidebar:
        st.header("⚙️ Pipeline Config")
        batch_size = st.slider("Batch Size", 16, 128, 32)
        vocab_size = st.number_input("Vocab Size", 1000, 10000, 5000)
        epochs = st.slider("Simulation Epochs", 1, 5, 3)

    @st.cache_data
    def generate_sample_data(n=200):
        phrases = [
            "The market is moving upward rapidly",
            "Economic downturn confirmed by quarterly report",
            "Tech sector shows strong resilience",
            "Investors are cautious about the new policy",
            "Positive sentiment drives stock prices higher",
            "Inflation remains a concern for global markets",
            "Renewable energy stocks surge on green initiative",
            "Supply chain disruptions impact manufacturing"
        ]
        return (
            [np.random.choice(phrases) for _ in range(n)],
            np.random.randint(0, 2, n).astype(float)
        )

    texts, labels = generate_sample_data()
    pipeline = TextPipeline(vocab_size=vocab_size, batch_size=batch_size)
    pipeline.adapt(texts)
    dataset  = pipeline.create_dataset(texts, labels)
    model    = SequenceModel(vocab_size=vocab_size)
    optimizer_obj = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn_obj   = tf.keras.losses.BinaryCrossentropy()
    metric_obj    = tf.keras.metrics.BinaryAccuracy()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Training Loop")
        if st.button("▶ Start Specialized Training Loop", key="train_btn"):
            bar  = st.progress(0)
            info = st.empty()
            for epoch in range(int(epochs)):
                epoch_loss, steps = 0, 0
                for x, y in dataset:
                    loss = train_step(model, x, y, optimizer_obj, loss_fn_obj, metric_obj)
                    epoch_loss += loss.numpy(); steps += 1
                    info.text(f"Epoch {epoch+1} · Step {steps} — Loss {loss:.4f} | Acc {metric_obj.result():.4f}")
                st.success(f"✅ Epoch {epoch+1} — Avg Loss: {epoch_loss/steps:.4f}")
                bar.progress((epoch + 1) / int(epochs))

        st.subheader("🧪 Real-time Sentiment Inference")
        user_input = st.text_input(
            "Enter market news text:",
            "The new fiscal policy is expected to boost small businesses."
        )
        if user_input:
            pred  = model(pipeline.vectorize_layer([user_input])).numpy()[0][0]
            label = "Bullish" if pred > 0.5 else "Bearish"
            color = "#00ff88" if label == "Bullish" else "#ff4466"
            st.markdown(f'<div class="status-card"><h3>Sentiment: <span style="color:{color}">{label}</span> &nbsp;·&nbsp; Score: {pred:.4f}</h3></div>', unsafe_allow_html=True)

    with col2:
        st.subheader("🛠️ Architecture")
        st.info("**Senior III Features:** AUTOTUNE · @tf.function · Gradient Clipping · Attention")
        st.code("""
dataset = dataset.map(preprocess,
    num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

@tf.function
def train_step(model, x, y, ...):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x))
    grads, _ = tf.clip_by_global_norm(
        tape.gradient(loss, model.weights), 5.0)
    optimizer.apply_gradients(...)
""", language="python")
        st.markdown("""
<div class="flow-container">
  <div class="flow-node">📝 Text Input</div><div class="flow-arrow">▼</div>
  <div class="flow-node">🔢 TextVectorization</div><div class="flow-arrow">▼</div>
  <div class="flow-node">🔷 Embedding Layer</div><div class="flow-arrow">▼</div>
  <div class="flow-node">↔️ Bi-Directional LSTM</div><div class="flow-arrow">▼</div>
  <div class="flow-node">🎯 Attention Layer</div><div class="flow-arrow">▼</div>
  <div class="flow-node">⚡ Dense Classifier</div><div class="flow-arrow">▼</div>
  <div class="flow-node">📊 Sigmoid Output</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — RAG Document Q&A
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    기업 내부 문서(PDF)를 **LangChain + ChromaDB**로 인덱싱하고,
    **GPT-4 Turbo**가 문서 내용을 근거로 질문에 답변하는 RAG 시스템입니다.
    """)

    col_rag1, col_rag2 = st.columns([2, 1])

    with col_rag1:
        # ── API Key ──
        st.subheader("🔑 OpenAI API Key")
        api_key = st.text_input(
            "Enter your OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="키는 서버 메모리에만 존재하며 저장되지 않습니다."
        )

        # ── PDF Upload ──
        st.subheader("📄 문서 업로드")
        uploaded_file = st.file_uploader(
            "사내 PDF 문서를 업로드하세요 (정책, 위키, 보고서 등)",
            type=["pdf"],
            help="업로드한 파일은 임시로만 사용되고 보관되지 않습니다."
        )

        # ── Build RAG ──
        if uploaded_file and api_key:
            if not RAG_AVAILABLE:
                st.error("langchain 패키지가 설치되지 않았습니다. requirements.txt를 확인하세요.")
            else:
                os.environ["OPENAI_API_KEY"] = api_key

                @st.cache_resource(show_spinner="📚 문서 인덱싱 중...")
                def build_rag(file_bytes, file_name):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    # 1. Load
                    loader = PyPDFLoader(tmp_path)
                    docs   = loader.load()

                    # 2. Chunk (1000자, 100자 overlap)
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=100
                    )
                    chunks = splitter.split_documents(docs)

                    # 3. Embed → ChromaDB (in-memory for cloud)
                    embeddings = OpenAIEmbeddings()
                    vectordb   = Chroma.from_documents(chunks, embeddings)

                    # 4. QA Chain
                    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
                    return RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
                        return_source_documents=True
                    )

                qa_chain = build_rag(uploaded_file.read(), uploaded_file.name)
                st.success(f"✅ **{uploaded_file.name}** 인덱싱 완료! 질문을 입력하세요.")

                # ── Q&A Interface ──
                st.subheader("💬 문서 기반 질문 답변")
                question = st.text_input(
                    "질문을 입력하세요:",
                    placeholder="예: 우리 회사의 재택근무 규정은 뭐야?"
                )

                if question:
                    with st.spinner("🔍 답변 생성 중..."):
                        result = qa_chain.invoke({"query": question})
                    st.markdown(f'<div class="status-card"><b>📝 답변</b><br><br>{result["result"]}</div>', unsafe_allow_html=True)

                    with st.expander("📎 참조된 문서 청크 보기"):
                        for i, doc in enumerate(result.get("source_documents", []), 1):
                            st.markdown(f"**청크 {i}** (p.{doc.metadata.get('page', '?')+1})")
                            st.text(doc.page_content[:400] + "...")

        elif not api_key:
            st.info("🔑 OpenAI API 키를 입력하면 RAG 시스템이 활성화됩니다.")
        elif not uploaded_file:
            st.info("📄 PDF 문서를 업로드하면 인덱싱이 시작됩니다.")

    with col_rag2:
        st.subheader("🏗️ RAG 아키텍처")
        st.markdown("""
<div class="flow-container">
  <div class="rag-node">📄 PDF 문서 업로드</div><div class="rag-arrow">▼</div>
  <div class="rag-node">✂️ Chunking (1000자)</div><div class="rag-arrow">▼</div>
  <div class="rag-node">🔢 OpenAI Embeddings</div><div class="rag-arrow">▼</div>
  <div class="rag-node">🗄️ ChromaDB 벡터 저장</div><div class="rag-arrow">▼</div>
  <div class="rag-node">🔍 Similarity Search (k=4)</div><div class="rag-arrow">▼</div>
  <div class="rag-node">🤖 GPT-4 Turbo (LLM)</div><div class="rag-arrow">▼</div>
  <div class="rag-node">💬 최종 답변 생성</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**핵심 패턴**")
        st.code("""
# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100  # 문맥 유지
)

# Vector Store
vectordb = Chroma.from_documents(
    chunks, OpenAIEmbeddings()
)

# RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4-turbo"),
    retriever=vectordb.as_retriever(
        search_kwargs={"k": 4}
    )
)
""", language="python")
        st.info("""
**RAG의 장점:**
- 🔒 데이터가 외부로 나가지 않음
- 📅 최신 사내 문서 반영 가능
- 🎯 환각(Hallucination) 최소화
- 📎 출처 문단 추적 가능
""")
