from pathlib import Path
import sys, json, random
import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import NameGenTransformer, NameGenLSTM
from src.safety import is_valid_name

st.set_page_config(page_title="NameGen", page_icon="🇮🇳", layout="wide")

@st.cache_resource
def load_assets():
    art = ROOT / "artifacts"
    vocab_data = json.loads((art / "vocab.json").read_text(encoding="utf-8"))
    stoi = vocab_data["stoi"]
    itos = {int(k): v for k, v in vocab_data["itos"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = NameGenTransformer(len(stoi)).to(device)
    transformer.load_state_dict(torch.load(art / "transformer.pth", map_location=device, weights_only=True))
    transformer.eval()

    lstm = NameGenLSTM(len(stoi)).to(device)
    lstm.load_state_dict(torch.load(art / "lstm.pth", map_location=device, weights_only=True))
    lstm.eval()
    return stoi, itos, transformer, lstm, device

stoi, itos, transformer, lstm, device = load_assets()
PAD, BOS, EOS, UNK = (stoi[t] for t in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"])

def sample_next(logits, temperature):
    logits = logits / max(float(temperature), 0.05)
    for token in [PAD, BOS, UNK]:
        logits[token] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def generate_one(model, kind, temperature=0.8, max_chars=24):
    ids = [BOS]
    with torch.no_grad():
        for _ in range(max_chars):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)
            next_id = sample_next(logits[0, -1, :].clone(), temperature)
            if next_id == EOS:
                break
            ids.append(next_id)
    chars = [itos[i] for i in ids[1:] if i not in (PAD, BOS, EOS, UNK)]
    return "".join(chars).strip()

def generate_batch(model, kind, count, temperature, max_chars):
    accepted, attempts = [], 0
    while len(accepted) < count and attempts < count * 3:
        attempts += 1
        name = generate_one(model, kind, temperature, max_chars)
        if is_valid_name(name) and name.lower() not in {x.lower() for x in accepted}:
            accepted.append(name.title())
    return accepted

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(115deg,#172554,#4338ca); color:white;}
.hero h1 {color:white; margin:0 0 .3rem 0;}
.hero p {color:#dbeafe; margin:0;}
.name-card {padding:.7rem .9rem; border:1px solid rgba(128,128,128,.28); border-radius:10px; margin:.35rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>NameGen</h1><p>Mini Transformer-Based Indian Place Name Generator</p></div>', unsafe_allow_html=True)
st.write("Generate Indian-style place names using character-level deep learning models trained on GeoNames India data.")
with st.sidebar:
    st.header("Generation settings")
    model_choice = st.selectbox("Choose model", ["Mini Transformer", "LSTM"])
    count = st.slider("Number of names", 1, 10, 5)
    temperature = st.slider("Temperature", 0.2, 1.5, 0.8, 0.1)
    max_chars = st.slider("Maximum generated characters", 5, 20, 12)
    st.caption("Lower temperature produces more predictable outputs; higher temperature increases randomness.")

active_model = transformer if model_choice == "Mini Transformer" else lstm
c1,c2,c3 = st.columns(3)
c1.metric("Selected model", model_choice)
c2.metric("Vocabulary size", len(stoi))
c3.metric("Device", str(device).upper())

left,right = st.columns([1.3,0.7], gap="large")
with left:
    st.subheader("Generate names")

    if st.button("✨ Generate names", type="primary", use_container_width=True):
        with st.spinner("Generating Indian place names..."):
            names = generate_batch(
                active_model,
                model_choice,
                count,
                temperature,
                max_chars
            )

        st.session_state.latest = names
        st.session_state.setdefault("history", [])
        st.session_state.history.insert(
            0,
            {
                "model": model_choice,
                "temperature": temperature,
                "names": names
            }
        )

        if not names:
            st.warning("No valid names generated. Try increasing temperature.")

    if "latest" in st.session_state:
        st.markdown("#### Generated results")
        for i, name in enumerate(st.session_state.latest, 1):
            st.markdown(
                f'<div class="name-card"><b>{i:02d}.</b> {name}</div>',
                unsafe_allow_html=True
            )

        if st.session_state.latest:
            st.download_button(
                "Download names CSV",
                pd.DataFrame({"place_name": st.session_state.latest}).to_csv(index=False),
                "namegen_results.csv",
                "text/csv"
            )
with right:
    st.subheader("About NameGen")
    st.write("A character-level model predicts one character at a time. The Transformer uses sinusoidal positional encoding and masked self-attention.")
    st.write("The LSTM is included as a baseline for comparison.")
    mascot = ROOT/"assets"/"mascot.png"
    if mascot.exists():
        st.image(str(mascot), use_container_width=True)
    else:
        st.info("Mascot artwork can be added here.")
    st.caption("Generated names are suggestions, not verified geographical locations.")

st.divider()
tab1,tab2 = st.tabs(["Generation history","Training metrics"])
with tab1:
    for i,item in enumerate(st.session_state.get("history",[]),1):
        with st.expander(f"Run {i} · {item['model']} · temperature {item['temperature']}"):
            st.write(", ".join(item["names"]) if item["names"] else "No valid names in this run.")
with tab2:
    metrics_path = ROOT/"artifacts"/"training_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)
        st.line_chart(metrics.set_index("epoch")[["transformer_train","transformer_val","lstm_train","lstm_val"]])
    loss_path = ROOT/"artifacts"/"training_loss.png"
    if loss_path.exists():
        st.image(str(loss_path), caption="Training and validation loss curves", use_container_width=True)

st.caption("Safety: generated outputs pass a basic format and blocklist filter. This is not comprehensive moderation.")
