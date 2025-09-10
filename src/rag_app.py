#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 00:20:01 2025

@author: operard
"""

import os
import streamlit as st
from rag_pdf_oracle_hybrid_search import rag_context, call_local_llm

# --- Streamlit page config ---
st.set_page_config(page_title="RAG Agent with Oracle + Local LLM", layout="wide")
st.title("🤖 RAG Agent (Oracle + Local LLM)")

# --- Sidebar config ---
st.sidebar.header("Configuration")

provider = st.sidebar.selectbox("Local LLM Provider", ["ollama", "lmstudio"])
model = st.sidebar.text_input("Model name", value="llama3.2")

default_k = int(os.getenv("DEFAULT_K", "6"))
k = st.sidebar.slider("Number of chunks (k)", min_value=1, max_value=20, value=default_k, step=1)

# --- Main input ---
query = st.text_area("Enter your question:", "")

if st.button("Run Query"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running RAG pipeline..."):
            try:
                # Step 1: Hybrid Search + Context
                bundle = rag_context(query, k=k)

                # Step 2: Call local LLM
                answer = call_local_llm(bundle["query"], bundle["context"], provider=provider, model=model)
                bundle["answer"] = answer

                # Step 3: Display results
                st.subheader("🤖 LLM Answer")
                st.write(answer)

                st.subheader("📄 Retrieved Context (concatenated)")
                st.text(bundle["context"])

                # Step 4: Display documents and hits
                st.subheader("📑 Hits by Document")
                hits = bundle["hits"]

                # Group by doc_id
                docs = {}
                for h in hits:
                    doc = h.get("doc_id", "Unknown")
                    docs.setdefault(doc, []).append(h)

                for doc_id, doc_hits in docs.items():
                    st.markdown(f"### 📂 Document: `{doc_id}`")
                    # Order by distance if available
                    doc_hits_sorted = sorted(doc_hits, key=lambda x: x.get("distance", 9999))
                    for i, h in enumerate(doc_hits_sorted, start=1):
                        try:
                            #st.markdown(
                            #    f"**Hit {i}:** Page {h.get('page')}, Chunk {h.get('chunk')} "
                            #    f"(Distance: {h.get('distance', 'n/a'):.4f})"
                            #)
                            st.markdown(
                                f"**Hit {i}:** Page {h.get('page')}, Chunk {h.get('chunk')} "
                                f"(Distance: {h.get('distance', 'n/a')})"
                            )
                            st.write(h.get("text", "").strip())
                        except Exception as e2:
                            st.error(f"Error loop: {e2}")

            except Exception as e:
                st.error(f"Error: {e}")
