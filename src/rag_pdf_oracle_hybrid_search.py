#!/usr/bin/env python3
"""
RAG PDF → Oracle (multilingual, hybrid search)

• Parses complex PDFs (layouts, tables, links, images w/ optional OCR)
• Intelligent, layout-aware chunking suitable for RAG
• Multilingual embeddings: switch between LOCAL (HuggingFace/SBERT) and OCI/Cohere-compatible endpoint
• Stores chunks + metadata + vectors in Oracle 23ai (AI Vector Search) with Oracle Text for BM25-ish lexical
• Hybrid retrieval (vector + lexical) with Reciprocal Rank Fusion

Tested on Python 3.10+ (macOS). External services optional.

──────────────────────────────────────────────────────────────────────────────
INSTALL

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install oracledb pdfplumber pymupdf pillow pytesseract blingfire
pip install sentence-transformers  # if EMBED_PROVIDER=local
pip install cohere                 # if EMBED_PROVIDER=oci (Cohere-compatible)

# For OCR (optional):
# • Install Tesseract on macOS: `brew install tesseract`

──────────────────────────────────────────────────────────────────────────────
ENV VARS

# DB (Autonomous DB or on-prem 23ai)
export ORA_DSN="adb.eu-frankfurt-1.oraclecloud.com:1522/your_tp_name"
export ORA_USER="YOUR_USER"
export ORA_PWD="YOUR_PASSWORD"

# Embeddings
# Option A: Local multilingual (default)
export EMBED_PROVIDER=local
export LOCAL_EMBED_MODEL=intfloat/multilingual-e5-large  # 1024-dim
export LOCAL_EMBED_MODEL=intfloat/multilingual-e5-base  # 768-dim

# Option B: OCI Generative AI or Cohere-compatible endpoint
# (Cohere SDK supports a base_url; OCI exposes Cohere-compatible endpoints.)
# Example:
# export EMBED_PROVIDER=oci
# export COHERE_API_KEY="<key with access to OCI/Cohere endpoint>"
# export COHERE_API_BASE="https://inference.generativeai.<region>.oci.oraclecloud.com/"  # if using OCI
# export OCI_EMBED_MODEL="cohere.embed-multilingual-v3.0"  # 1024-dim

──────────────────────────────────────────────────────────────────────────────
USAGE

# 1) Ingest PDFs from a folder into Oracle (creates tables + indexes if needed)
python rag_pdf_oracle.py ingest ./pdfs

# 2) Ask a question (hybrid search over your corpus)
python rag_pdf_oracle.py query "¿Qué dice el manual sobre la traza de auditoría?"

# 3) Or export matches to JSON for use in a separate LLM step
python rag_pdf_oracle.py search "hitos y objetivos" --k 8 --out hits.json

──────────────────────────────────────────────────────────────────────────────
NOTES
- Requires Oracle Database 23ai (VECTOR type + VECTOR INDEX) and Oracle Text enabled.
- If your DB is older, you can store vectors as BLOB and skip ANN; adjust SQL accordingly.
- The chunker keeps tables as Markdown and tracks page numbers, section titles, and links.
"""
from __future__ import annotations
import os
import sys
import io
import json
import time
import math
import hashlib
import array
import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional, Tuple

import tempfile  # Para archivos temporales
from oci_pdf_lister import OCIPDFObjectStorageLister  # Importa la librería que creamos


# ----------------------------
# Embedding providers
# ----------------------------
class Embedder:
    dim: int
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class LocalHFEmbedder(Embedder):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Common multilingual dims: e5-large=1024, mpnet-multilingual=768
        self.dim = 1024 if "e5-large" in model_name else 768

    def _prep(self, t: str) -> str:
        # e5 models expect "query: ..." / "passage: ..." formats; we'll use passage for indexing
        return f"passage: {t.strip()}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        enc = self.model.encode([self._prep(t) for t in texts],
                                convert_to_numpy=True,
                                normalize_embeddings=True)
        return enc.tolist()

class OCIOrCohereEmbedder(Embedder):
    """Cohere-compatible client (works with Cohere cloud or OCI Generative AI compatible endpoints).
    Set COHERE_API_KEY and optionally COHERE_API_BASE.
    Model examples: cohere.embed-multilingual-v3.0 (1024-dim)
    """
    def __init__(self, model: str = "cohere.embed-multilingual-v3.0"):
        import cohere
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is required for EMBED_PROVIDER=oci")
        base = os.getenv("COHERE_API_BASE")  # Optional. If set to OCI endpoint, SDK routes there.
        self.client = cohere.ClientV2(api_key=api_key, base_url=base) if hasattr(cohere, 'ClientV2') else cohere.Client(api_key, base_url=base)
        self.model = model
        self.dim = 1024

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Cohere v5 SDK (ClientV2) uses .embeddings.create; older uses .embed
        try:
            resp = self.client.embeddings.create(model=self.model, input_texts=texts, input_type="search_document", truncate="RIGHT")
            vecs = [e.embedding for e in resp.embeddings]
        except Exception:
            # fallback older SDK method name
            resp = self.client.embed(model=self.model, texts=texts, input_type="search_document")
            vecs = resp.embeddings
        # Normalize for cosine
        def l2n(v):
            s = math.sqrt(sum(x*x for x in v)) or 1.0
            return [x/s for x in v]
        return [l2n(v) for v in vecs]


def get_embedder() -> Embedder:
    provider = os.getenv("EMBED_PROVIDER", "local").lower()
    if provider == "local":
        model = os.getenv("LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-large")
        return LocalHFEmbedder(model)
    elif provider == "oci":
        model = os.getenv("OCI_EMBED_MODEL", "cohere.embed-multilingual-v3.0")
        return OCIOrCohereEmbedder(model)
    else:
        raise RuntimeError(f"Unknown EMBED_PROVIDER: {provider}")

# ----------------------------
# Oracle database helpers
# ----------------------------
import oracledb
import decimal

def normalize_row(row):
    """Convert Oracle row values into JSON-safe Python types."""
    safe = []
    for v in row:
        if isinstance(v, oracledb.LOB):
            safe.append(v.read())   # convert LOB → str
        elif isinstance(v, decimal.Decimal):
            # if it’s an integer-like value, cast to int; else float
            safe.append(int(v) if v == v.to_integral_value() else float(v))
        else:
            safe.append(v)
    return tuple(safe)

def normalize_rows(rows):
    return [normalize_row(r) for r in rows]




SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS RAG_CHUNKS (
  ID           VARCHAR2(64) PRIMARY KEY,
  DOC_ID       VARCHAR2(200),
  PAGE_NO      NUMBER,
  CHUNK_ORD    NUMBER,
  TEXT         CLOB,
  METADATA     JSON,
  EMBEDDING    VECTOR(:dim)
)
"""

SCHEMA_INDEX_SQL = [
    # Oracle Text index for lexical search
    "CREATE INDEX IF NOT EXISTS RAG_CHUNKS_TEXT_IDX ON RAG_CHUNKS(TEXT) INDEXTYPE IS CTXSYS.CONTEXT",
    # ANN index for vector search (choose IVF as a good default; HNSW is also available)
    "CREATE VECTOR INDEX IF NOT EXISTS RAG_CHUNKS_VEC_IDX ON RAG_CHUNKS (EMBEDDING) ORGANIZATION IVF WITH (DISTANCE = COSINE)"
]

DROP_INDEX_SQL = [
    "DROP INDEX RAG_CHUNKS_TEXT_IDX",
    "DROP INDEX RAG_CHUNKS_VEC_IDX FORCE"
]

UPSERT_SQL = """
MERGE INTO RAG_CHUNKS t
USING (
  SELECT :id AS id, :doc_id AS doc_id, :page_no AS page_no, :chunk_ord AS chunk_ord,
         :text AS text, :metadata AS metadata, :embedding AS embedding
  FROM dual
) s
ON (t.ID = s.ID)
WHEN MATCHED THEN UPDATE SET t.DOC_ID=s.DOC_ID, t.PAGE_NO=s.PAGE_NO, t.CHUNK_ORD=s.CHUNK_ORD, t.TEXT=s.TEXT, t.METADATA=s.METADATA, t.EMBEDDING=s.EMBEDDING
WHEN NOT MATCHED THEN INSERT (ID, DOC_ID, PAGE_NO, CHUNK_ORD, TEXT, METADATA, EMBEDDING)
VALUES (s.ID, s.DOC_ID, s.PAGE_NO, s.CHUNK_ORD, s.TEXT, s.METADATA, s.EMBEDDING)
"""

INSERT_SQL = """
INSERT INTO RAG_CHUNKS (ID, DOC_ID, PAGE_NO, CHUNK_ORD, TEXT, METADATA, EMBEDDING)
VALUES (:id, :doc_id, :page_no, :chunk_ord, :text, :metadata, :embedding)
"""

DELETE_DOC_SQL = """
DELETE FROM RAG_CHUNKS WHERE DOC_ID = :doc_id
"""

VSEARCH_SQL = """
-- Vector candidates\n SELECT ID, DOC_ID, PAGE_NO, CHUNK_ORD, TEXT, METADATA,
       VECTOR_DISTANCE(EMBEDDING, :qvec, COSINE) AS VDIST
  FROM RAG_CHUNKS
 ORDER BY VDIST ASC
 FETCH FIRST :k ROWS ONLY
"""

LSEARCH_SQL = """
-- Lexical candidates with Oracle Text
SELECT ID, DOC_ID, PAGE_NO, CHUNK_ORD, TEXT, METADATA, SCORE(1) AS LScore
  FROM RAG_CHUNKS
 WHERE CONTAINS(TEXT, :q, 1) > 0
 ORDER BY SCORE(1) DESC
 FETCH FIRST :k ROWS ONLY
"""

# ----------------------------
# PDF parsing (layout-aware) + chunking
# ----------------------------

@dataclass
class Chunk:
    doc_id: str
    page_no: int
    chunk_ord: int
    text: str
    meta: Dict[str, Any]


def hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:64]


def safe_read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def extract_links(page) -> List[Dict[str, Any]]:
    links = []
    try:
        for l in page.get_links():
            if l.get("uri"):
                # links.append({"type": "uri", "uri": l["uri"], "from": l.get("from")})
                rect = l.get("from")
                if rect is not None:
                    rect = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                links.append({"type": "uri", "uri": l["uri"], "from": rect})
            elif l.get("kind") == 2 and l.get("page") is not None:
                # links.append({"type": "internal", "page": int(l["page"])})
                rect = l.get("from")
                if rect is not None:
                    rect = [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                links.append({"type": "internal", "page": int(l["page"]), "from": rect})
                
    except Exception:
        pass
    return links


def extract_tables_pdfplumber(pdf_path: str) -> Dict[int, List[str]]:
    """Return per-page list of Markdown tables (pipe format)."""
    import pdfplumber
    tables_by_page: Dict[int, List[str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            md_tables = []
            try:
                tables = page.extract_tables()
                for tbl in tables or []:
                    if not tbl or not any(any(cell for cell in row) for row in tbl):
                        continue
                    # convert to Markdown pipe table
                    # normalize row lengths
                    maxc = max(len(r) for r in tbl)
                    norm = [[(c or "").strip() for c in r] + [""]*(maxc-len(r)) for r in tbl]
                    header = norm[0]
                    sep = ["---"] * maxc
                    rows = norm[1:]
                    lines = ["| " + " | ".join(header) + " |",
                             "| " + " | ".join(sep) + " |"]
                    for r in rows:
                        lines.append("| " + " | ".join(r) + " |")
                    md_tables.append("\n".join(lines))
            except Exception:
                pass
            if md_tables:
                tables_by_page[pno] = md_tables
    return tables_by_page


def image_ocr_from_page(doc, page, ocr: bool) -> List[str]:
    """Extract large images from page and OCR if enabled; return markdown blocks with alt text."""
    import pytesseract
    from PIL import Image
    md = []
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.width * pix.height < 400*400:  # skip tiny assets
                continue
            mode = "RGB" if pix.n < 4 else "RGBA"
            pil = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            alt = ""
            if ocr:
                try:
                    alt = pytesseract.image_to_string(pil)
                except Exception:
                    alt = ""
            # save image? not necessary for DB; we inline as a placeholder text
            md.append(f"![figure p{page.number+1}](image@{xref})\n\n{alt.strip()}")
    except Exception:
        pass
    return md


def detect_headings(page_dict: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Return list of (text, fontsize) for potential headings (heuristic: top font sizes)."""
    heads = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = float(span.get("size", 0.0))
                if text and len(text) < 140 and text.isupper() or size >= 12.5:
                    heads.append((text, size))
    # Keep unique-ish in order, prefer bigger sizes
    uniq = []
    seen = set()
    for t,s in sorted(heads, key=lambda x: -x[1]):
        if t not in seen:
            seen.add(t)
            uniq.append((t,s))
    return uniq[:10]


def split_sentences(text: str) -> List[str]:
    """Multilingual sentence split using blingfire if available; fallback to regex."""
    try:
        import blingfire as bf
        return [s.strip() for s in bf.text_to_sentences(text).split("\n") if s.strip()]
    except Exception:
        # simple fallback
        import re
        parts = re.split(r"(?<=[\.!?])\s+", text.strip())
        return [p for p in parts if p]


def chunk_paragraphs(paragraphs: List[str], max_chars=1200, overlap_sents=1) -> List[str]:
    chunks = []
    buf = []
    length = 0
    for para in paragraphs:
        sents = split_sentences(para)
        for s in sents:
            if length + len(s) + 1 > max_chars and buf:
                chunks.append(" ".join(buf).strip())
                # overlap
                buf = buf[-overlap_sents:]
                length = sum(len(x) for x in buf)
            buf.append(s)
            length += len(s) + 1
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def parse_pdf_to_chunks(pdf_path: str, doc_id: Optional[str] = None, ocr_images=False, bTemp=False) -> List[Chunk]:
    import fitz  # PyMuPDF
    tables_by_page = extract_tables_pdfplumber(pdf_path)
    data = safe_read(pdf_path)
    with fitz.open(stream=data, filetype="pdf") as doc:
        chunks: List[Chunk] = []
        doc_id = doc_id or os.path.basename(pdf_path)
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            page_dict = page.get_text("dict")
            page_text = page.get_text("text")
            links = extract_links(page)
            headings = detect_headings(page_dict)

            # Split page text into layout paragraphs (by blocks) to keep structure
            paras: List[str] = []
            for block in page.get_text("blocks"):
                # block = (x0, y0, x1, y1, text, block_no, block_type, ...) depending on PyMuPDF ver
                if len(block) >= 5 and isinstance(block[4], str):
                    t = block[4].strip()
                    if t:
                        paras.append(t)

            # Tables as separate markdown paragraphs
            if pno+1 in tables_by_page:
                for md in tables_by_page[pno+1]:
                    paras.append("\nTable:\n" + md)

            # Large images (optional OCR)
            for md in image_ocr_from_page(doc, page, ocr=ocr_images):
                paras.append(md)

            # If no paras captured, fallback to whole page text
            if not paras and page_text.strip():
                paras = [page_text.strip()]

            # Chunking
            text_chunks = chunk_paragraphs(paras, max_chars=1200, overlap_sents=1)

            # Metadata common to page
            if bTemp:
                page_meta = {
                    "page": pno+1,
                    "links": links,
                    "headings_top": [h for h,_ in headings],
                    "source_pdf": doc_id,
                }
            else:
                page_meta = {
                    "page": pno+1,
                    "links": links,
                    "headings_top": [h for h,_ in headings],
                    "source_pdf": os.path.basename(pdf_path),
                }
                
            for i, t in enumerate(text_chunks):
                meta = dict(page_meta)
                meta["chunk_ord"] = i
                chunk_id = hash_id(f"{doc_id}|{pno+1}|{i}|{hashlib.md5(t.encode()).hexdigest()}")
                chunks.append(Chunk(doc_id=doc_id, page_no=pno+1, chunk_ord=i, text=t, meta=meta))
        return chunks

# ----------------------------
# Ingest → Oracle
# ----------------------------

def connect_oracle():
    dsn = os.getenv("ORA_DSN")
    user = os.getenv("ORA_USER")
    pwd = os.getenv("ORA_PWD")
    if not all([dsn, user, pwd]):
        raise RuntimeError("Set ORA_DSN, ORA_USER, ORA_PWD env vars")
    return oracledb.connect(user=user, password=pwd, dsn=dsn)


def ensure_schema(conn, dim: int):
    with conn.cursor() as cur:
        ddl = f"""
        CREATE TABLE RAG_CHUNKS (
          ID           VARCHAR2(64) PRIMARY KEY,
          DOC_ID       VARCHAR2(200),
          PAGE_NO      NUMBER,
          CHUNK_ORD    NUMBER,
          TEXT         CLOB,
          METADATA     JSON,
          EMBEDDING    VECTOR({dim}, FLOAT32)
        )
        """
        try:
            cur.execute(ddl)
        except oracledb.DatabaseError as e:
            if "ORA-00955" not in str(e):  # ignore "name is already used by existing object"
                raise
        conn.commit()


def create_indexes(conn):
    with conn.cursor() as cur:
        for stmt in SCHEMA_INDEX_SQL:
            try:
                cur.execute(stmt)
            except oracledb.DatabaseError as e:
                if "ORA-" in str(e):
                    pass
        conn.commit()


def drop_indexes(conn):
    with conn.cursor() as cur:
        for stmt in DROP_INDEX_SQL:
            try:
                cur.execute(stmt)
            except oracledb.DatabaseError as e:
                if "ORA-01418" in str(e):  # index does not exist
                    pass
                else:
                    raise
        conn.commit()


def ingest_pdfs(folder: str, ocr_images=False):
    embedder = get_embedder()
    conn = connect_oracle()
    ensure_schema(conn, embedder.dim)
    drop_indexes(conn)  # Drop indexes to speed up inserts

    pdf_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".pdf") and not f.startswith(".")
    ]

    if not pdf_paths:
        print("No PDFs found in", folder)
        return

    with conn.cursor() as cur:
        for pdf in pdf_paths:
            print(f"→ Parsing {pdf}")
            chunks = parse_pdf_to_chunks(pdf, doc_id=os.path.basename(pdf), ocr_images=ocr_images, bTemp=False)
            if not chunks:
                continue
            texts = [c.text for c in chunks]
            print(f"   {len(texts)} chunks → embedding…")
            vecs = embedder.embed_texts(texts)

            # Delete existing chunks for this doc_id to avoid duplicates and use fast INSERT
            cur.execute(DELETE_DOC_SQL, dict(doc_id=chunks[0].doc_id))
            conn.commit()

            # Prepare batch params
            params_list = []
            for c, v in zip(chunks, vecs):
                params_list.append(dict(
                    id=hash_id(f"{c.doc_id}|{c.page_no}|{c.chunk_ord}"),
                    doc_id=c.doc_id,
                    page_no=c.page_no,
                    chunk_ord=c.chunk_ord,
                    text=c.text,
                    metadata=json.dumps(c.meta, ensure_ascii=False, default=str),
                    embedding=array.array('f', v),
                ))

            # Batch insert
            cur.executemany(INSERT_SQL, params_list)
            conn.commit()
            print(f"   Stored {len(chunks)} chunks into Oracle")

    create_indexes(conn)  # Recreate indexes after all inserts


def ingestfromoci(compartment_id, bucket_name, namespace_name, prefix="", ocr_images=False):
    """
    Ingests PDFs from OCI Object Storage instead of a local folder.
    
    :param compartment_id: ID del compartment de OCI.
    :param bucket_name: Nombre del bucket.
    :param namespace_name: Namespace del tenant.
    :param prefix: Prefijo del directorio en el bucket (e.g., "documentos/"). Deja vacío para el root.
    :param ocr_images: Parámetro para OCR (igual que antes).
    """
    embedder = get_embedder()
    conn = connect_oracle()
    ensure_schema(conn, embedder.dim)
    drop_indexes(conn)  # Drop indexes to speed up inserts

    # Crea la instancia de la librería para acceder a OCI Object Storage
    lister = OCIPDFObjectStorageLister(compartment_id, bucket_name, namespace_name)

    # Lista los PDFs en el prefijo especificado
    pdf_names = lister.list_pdf_files(prefix=prefix)

    if not pdf_names:
        print(f"No PDFs found in {bucket_name} under prefix '{prefix}'")
        return

    with conn.cursor() as cur:
        for pdf_name in pdf_names:
            print(f"→ Parsing {pdf_name}")
            
            # Descarga el PDF a un archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_path = temp_file.name
                success = lister.download_pdf(pdf_name, local_file_path=temp_path)
                
                if not success:
                    print(f"   Failed to download {pdf_name}, skipping...")
                    continue
            
            try:
                # Procesa el PDF usando la ruta temporal
                chunks = parse_pdf_to_chunks(temp_path, doc_id=pdf_name, ocr_images=ocr_images, bTemp=True)
                if not chunks:
                    continue
                texts = [c.text for c in chunks]
                print(f"   {len(texts)} chunks → embedding…")
                vecs = embedder.embed_texts(texts)

                # Delete existing chunks for this doc_id to avoid duplicates and use fast INSERT
                cur.execute(DELETE_DOC_SQL, dict(doc_id=chunks[0].doc_id))
                conn.commit()

                # Prepare batch params
                params_list = []
                for c, v in zip(chunks, vecs):
                    params_list.append(dict(
                        id=hash_id(f"{c.doc_id}|{c.page_no}|{c.chunk_ord}"),
                        doc_id=c.doc_id,
                        page_no=c.page_no,
                        chunk_ord=c.chunk_ord,
                        text=c.text,
                        metadata=json.dumps(c.meta, ensure_ascii=False, default=str),
                        embedding=array.array('f', v),
                    ))

                # Batch insert
                cur.executemany(INSERT_SQL, params_list)
                conn.commit()
                print(f"   Stored {len(chunks)} chunks into Oracle")
            
            finally:
                # Elimina el archivo temporal para liberar espacio
                os.unlink(temp_path)

    create_indexes(conn)  # Recreate indexes after all inserts

# ----------------------------
# Hybrid search (vector + lexical) with RRF
# ----------------------------

def rrf_merge(vec_rows, lex_rows, k=10) -> List[Dict[str, Any]]:
    # Reciprocal Rank Fusion
    ranks: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(vec_rows, start=1):
        rid = r[0]; ranks[rid] = ranks.get(rid, 0.0) + 1.0/(60+i)  # k=60 default constant
        items[rid] = r
    for j, r in enumerate(lex_rows, start=1):
        rid = r[0]; ranks[rid] = ranks.get(rid, 0.0) + 1.0/(60+j)
        items[rid] = r
    ordered = sorted(items.values(), key=lambda r: -ranks[r[0]])[:k]
    out = []
    for r in ordered:
        id, doc_id, page_no, chunk_ord, text, meta, extra = r
        out.append({
            "id": id,
            "doc_id": doc_id,
            "page": int(page_no),
            "chunk": int(chunk_ord),
            "text": text,
            "metadata": json.loads(meta) if isinstance(meta, str) else meta,
        })
    return out


def semantic_search(conn, table_name: str, collection: str, query_vec: List[float], top_k: int = 8):
    cur = conn.cursor()
    sql = f"""
        SELECT id, doc_id, source_path, page_num, chunk_id, section_path, urls_json, content,
               embedding <-> :qvec AS distance
        FROM {table_name}
        WHERE collection = :coll
        ORDER BY embedding <-> :qvec
        FETCH FIRST :k ROWS ONLY
    """
    cur.execute(sql, qvec=query_vec, coll=collection, k=top_k)
    cols = [d[0].lower() for d in cur.description]
    rows_raw = cur.fetchall()
    rows = [dict(zip(cols, normalize_row(r))) for r in rows_raw]
    return rows


def hybrid_search(query: str, k: int = 8) -> List[Dict[str, Any]]:
    embedder = get_embedder()
    qvec = embedder.embed_texts([query])[0]
    conn = connect_oracle()
    with conn.cursor() as cur:
        # Vector search
        cur.execute(VSEARCH_SQL, dict(qvec=array.array('f', qvec), k=k))
        # vec_rows = cur.fetchall()
        vec_rows = normalize_rows(cur.fetchall())
        # Lexical search
        # cur.execute(LSEARCH_SQL, dict(q=query, k=k))
        safe_q = f"{{ {query} }}"   # Oracle Text "simple text" literal mode
        cur.execute(LSEARCH_SQL, dict(q=safe_q, k=k))
        # lex_rows = cur.fetchall()
        lex_rows = normalize_rows(cur.fetchall())

    return rrf_merge(vec_rows, lex_rows, k=k)

# ----------------------------
# Simple RAG assemble (no generation)
# ----------------------------

def rag_context(query: str, k: int = 6, max_chars: int = 3000) -> Dict[str, Any]:
    hits = hybrid_search(query, k=k)
    ctx = []
    total = 0
    for h in hits:
        t = h["text"].strip()
        if total + len(t) > max_chars:
            break
        ctx.append(f"[p.{h['page']:>3}] {t}")
        total += len(t)
    return {"query": query, "context": "\n\n".join(ctx), "hits": hits}


# ----------------------------
# call LLM Model
# ----------------------------


import requests

def call_local_llm(query: str, context: str, provider: str = "ollama", model: str = "llama3"):
    """
    Call a local LLM (Ollama or LM Studio) with OpenAI-compatible API.
    provider: "ollama" | "lmstudio"
    model: model name (must exist in Ollama or LM Studio)
    """
    prompt = f"""Use the following context to answer the question:

{context}

Question: {query}
Answer:"""

    if provider == "ollama":
        url = "http://localhost:11434/v1/chat/completions"  # Ollama OpenAI-compatible endpoint
    elif provider == "lmstudio":
        url = "http://localhost:1234/v1/chat/completions"   # LM Studio default endpoint
    else:
        raise ValueError("Provider must be 'ollama' or 'lmstudio'")

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    
    
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ----------------------------
# CLI
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_pdf_oracle_hybrid_search.py <ingest|ingestfromoci|search|query> [args…]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python rag_pdf_oracle_hybrid_search.py ingest <pdf_folder> [--ocr]")
            sys.exit(1)
        folder = sys.argv[2]
        ocr = "--ocr" in sys.argv
        ingest_pdfs(folder, ocr_images=ocr)
    elif cmd == "ingestfromoci":
        if len(sys.argv) < 3:
            print("Usage: python rag_pdf_oracle_hybrid_search.py ingestfromoci <compartment_id> <bucket_name> <namespace_name> <prefix> [--ocr]")
            sys.exit(1)            
        compartment_id = sys.argv[2]
        bucket_name = sys.argv[3]
        namespace_name = sys.argv[4]
        prefix = sys.argv[5]  # Opcional: directorio en el bucket
        ocr = "--ocr" in sys.argv
        ingestfromoci(compartment_id, bucket_name, namespace_name, prefix=prefix, ocr_images=ocr)
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: python rag_pdf_oracle_hybrid_search.py search <query> [--k 8] [--out file.json]")
            sys.exit(1)
        query = sys.argv[2]
        k = 8
        if "--k" in sys.argv:
            k = int(sys.argv[sys.argv.index("--k")+1])
        results = hybrid_search(query, k=k)
        if "--out" in sys.argv:
            out = sys.argv[sys.argv.index("--out")+1]
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"Saved to {out}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Usage: python rag_pdf_oracle_hybrid_search.py query <question> [--k 6]")
            sys.exit(1)
        question = sys.argv[2]
        k = 6
        if "--k" in sys.argv:
            k = int(sys.argv[sys.argv.index("--k")+1])
        bundle = rag_context(question, k=k)
        # ---------------
        # Call LLM
        # ---------------
        # Choose local LLM provider
        provider = os.getenv("LOCAL_LLM_PROVIDER", "ollama")   # or "lmstudio"
        model = os.getenv("LOCAL_LLM_MODEL", "llama3")
        
        answer = call_local_llm(bundle["query"], bundle["context"], provider=provider, model=model)
        
        bundle["answer"] = answer

        print(json.dumps(bundle, ensure_ascii=False, indent=2, default=str))
                
    else:
        print("Unknown command:", cmd)
        sys.exit(1)

if __name__ == "__main__":
    # Late import to avoid global cost when not ingesting
    import fitz  # noqa: F401 (PyMuPDF)
    main()