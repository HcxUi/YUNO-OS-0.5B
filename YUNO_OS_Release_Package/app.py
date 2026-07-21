import sys
import os
import io
import numpy as np

# Fix NumPy 2.x compatibility before importing transformers
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# Safely ensure UTF-8 encoding if buffer is valid
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Page setup
st.set_page_config(
    page_title="YUNO OS - Intelligent Hybrid AI Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #171b26 0%, #0d1117 100%);
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #A855F7 0%, #EC4899 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 25px;
    }
    
    .badge {
        display: inline-block;
        padding: 5px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 12px;
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    .stChatMessage {
        background-color: rgba(22, 27, 34, 0.85);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        backdrop-filter: blur(10px);
    }
    
    .thought-container {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #8b5cf6;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "extracted_model")

def parse_reasoning_and_answer(raw_text):
    """
    Parses reasoning <think>...</think> block and clean answer from model output.
    """
    text = raw_text.strip()
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>")
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return thinking, answer
    elif "</think>" in text:
        parts = text.split("</think>")
        return parts[0].strip(), parts[1].strip()
    else:
        lines = text.split("\n")
        if lines and any(lines[0].strip().startswith(prefix) for prefix in [
            "The user", "I need to", "I should", "Let me", "The prompt", "Okay, the user"
        ]):
            for i, line in enumerate(lines):
                if line.strip() == "" and i > 0:
                    return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
        return "", text

@st.cache_resource(show_spinner="⏳ Unpacking & Loading YUNO OS Protected Binary Container...")
def load_yuno_model():
    if not os.path.exists(MODEL_DIR) or not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
        try:
            from extract_and_run import extract_yuno_bin, MODEL_FILE
            target_path = MODEL_FILE if os.path.exists(MODEL_FILE) else "yuno-os-v0.5.0.yuno"
            extract_yuno_bin(target_path, MODEL_DIR)
        except Exception as e:
            st.error(f"Failed to extract container: {e}")
            st.stop()
    
    config = AutoConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        config=config, 
        torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model

# Sidebar Configuration
with st.sidebar:
    st.markdown("<h2 style='color:#f0f6fc; font-family: Outfit;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    system_prompt = st.text_area(
        "System Instructions",
        value="You are YUNO OS, an intelligent hybrid AI operating system assistant created and developed by @hcxui. You respond clearly, accurately, and helpfully in both English and Hindi. If anyone asks who created, built, or developed you, state clearly and proudly that you were created by @hcxui.",
        height=130
    )
    
    show_thinking = st.toggle("🧠 Show Thought Process (<think>)", value=True)
    
    st.markdown("---")
    st.markdown("<h4 style='color:#f0f6fc;'>Hyperparameters</h4>", unsafe_allow_html=True)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.6, step=0.05)
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=1.3, value=1.08, step=0.01)
    max_new_tokens = st.slider("Max Tokens", min_value=64, max_value=1024, value=512, step=32)
    
    st.markdown("---")
    st.markdown("<h4 style='color:#f0f6fc;'>System Specifications</h4>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Creator: @hcxui</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Qwen3 Hybrid Engine</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>28 Layer Transformer</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>151.6k Vocab Size</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>40k Context Window</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>Bilingual English & Hindi</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header
st.markdown("<div class='main-header'>YUNO OS</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Next-Generation Intelligent Hybrid AI Assistant — Developed by <b>@hcxui</b></div>", unsafe_allow_html=True)

# Load model & tokenizer
tokenizer, model = load_yuno_model()

# Chat state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thinking") and show_thinking:
            with st.expander("💭 YUNO OS Thought Process", expanded=False):
                st.markdown(f"```text\n{msg['thinking']}\n```")
        st.markdown(msg["content"])

# User Input Box
if prompt := st.chat_input("Ask YUNO OS anything (in English or Hindi)..."):
    # Display User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Build ChatML Prompt with <think> tag
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for m in st.session_state.messages:
        full_prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    full_prompt += "<|im_start|>assistant\n<think>\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🌌 YUNO OS is thinking..."):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            res_tokens = outputs[0][inputs.input_ids.shape[1]:]
            raw_text = tokenizer.decode(res_tokens, skip_special_tokens=False)
            if raw_text.endswith("<|im_end|>"):
                raw_text = raw_text[:-10]
                
            thinking, answer = parse_reasoning_and_answer("<think>\n" + raw_text)
            
            # Display thought process if toggle active
            if thinking and show_thinking:
                with st.expander("💭 YUNO OS Thought Process", expanded=True):
                    st.markdown(f"```text\n{thinking}\n```")
                    
            message_placeholder.markdown(answer)
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "thinking": thinking
    })
