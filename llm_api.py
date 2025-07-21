import re
import json
import pandas as pd
from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["API_KEY"])

def ask_gemma(
    question: str,
    df: pd.DataFrame,
    max_rows: int = 20,
    model: str = "gpt-4.1-mini-2025-04-14",
) -> str:
    """
    Parse tickers, Category, and Secondary Category out of `question`
    and only send those rows to the model. If none are found, falls
    back to the first `max_rows` of df.
    """
    # 1) Gather all valid values from df
    all_tickers    = set(df["ETF"].astype(str))
    all_cats       = df["Category"].dropna().astype(str).unique().tolist()
    all_sec_cats   = df["Secondary Category"].dropna().astype(str).unique().tolist()

    # 2) Parse tickers: all-caps alphanum tokens of length 2–5
    candidates = re.findall(r"\b[A-Z0-9]{2,5}\b", question)
    tickers = [t for t in candidates if t in all_tickers]

    # 3) Parse Category & Secondary Category by substring match (case‑insensitive)
    categories = [
        cat for cat in all_cats
        if re.search(rf"\b{re.escape(cat)}\b", question, flags=re.IGNORECASE)
    ]
    sec_categories = [
        sc for sc in all_sec_cats
        if re.search(rf"\b{re.escape(sc)}\b", question, flags=re.IGNORECASE)
    ]

    df2 = df
    if tickers:
        df2 = df2[df2["ETF"].isin(tickers)]
    if categories:
        df2 = df2[df2["Category"].isin(categories)]
    if sec_categories:
        df2 = df2[df2["Secondary Category"].isin(sec_categories)]

    if df2.empty:
        df2 = df.head(max_rows)
    else:
        df2 = df2.head(max_rows)

    data_json = df2.to_json(orient="records", date_format="iso")

    messages = [
        {"role": "system", "content": "You are a professional analyst in the Canadian ETFs Industry. You will try your best to answer the questions using the information provided to you and your own domain knowledge。"},
        {"role": "user",   "content": f"Data slice:\n{data_json}\n\nQ: {question}"}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,    # type: ignore
        temperature=0.0,
        max_tokens=512,
    )
    if not resp.choices:
        raise RuntimeError(f"No response: {resp}")
    return resp.choices[0].message.content.strip()


def ask_gemma_with_context(
    question: str,
    conversation_history: list,
    df: pd.DataFrame = None,
    max_rows: int = 500,
    model: str = "gpt-4.1-nano-2025-04-14",
) -> str:
    """
    Context-aware version that maintains conversation history and only sends
    data when needed (first question or when context refresh is requested).
    """
    # Check if this is the first question or if data context is needed
    needs_data_context = (
        len(conversation_history) == 0 or  # First question
        "refresh" in question.lower() or   # User wants fresh context
        any("data" in msg.get("content", "").lower() for msg in conversation_history[-2:])  # Recent data reference
    )
    
    # Build messages starting with system prompt
    messages = [
        {"role": "system", "content": "You are a data‑savvy assistant for Canadian ETFs. When provided with data context, use it to answer questions. For follow-up questions, refer to the previously provided data context."}
    ]
    
    # Add conversation history
    messages.extend(conversation_history)
    
    # If we need data context, include relevant data
    if needs_data_context and df is not None:
        # Parse the question to find relevant data subset
        all_tickers = set(df["ETF"].astype(str))
        all_cats = df["Category"].dropna().astype(str).unique().tolist()
        all_sec_cats = df["Secondary Category"].dropna().astype(str).unique().tolist()

        # Parse for specific entities mentioned
        candidates = re.findall(r"\b[A-Z0-9]{2,5}\b", question)
        tickers = [t for t in candidates if t in all_tickers]
        
        categories = [
            cat for cat in all_cats
            if re.search(rf"\b{re.escape(cat)}\b", question, flags=re.IGNORECASE)
        ]
        sec_categories = [
            sc for sc in all_sec_cats
            if re.search(rf"\b{re.escape(sc)}\b", question, flags=re.IGNORECASE)
        ]

        # Filter data based on parsed entities
        df_filtered = df
        if tickers:
            df_filtered = df_filtered[df_filtered["ETF"].isin(tickers)]
        if categories:
            df_filtered = df_filtered[df_filtered["Category"].isin(categories)]
        if sec_categories:
            df_filtered = df_filtered[df_filtered["Secondary Category"].isin(sec_categories)]

        # If no specific entities found, use a representative sample
        if df_filtered.empty or (not tickers and not categories and not sec_categories):
            df_filtered = df.head(max_rows)
        else:
            df_filtered = df_filtered.head(max_rows)

        data_json = df_filtered.to_json(orient="records", date_format="iso")
        
        # Add data context to the conversation
        messages.append({
            "role": "user", 
            "content": f"Here is the ETF data context for our conversation:\n{data_json}\n\nQuestion: {question}"
        })
    else:
        # Just add the question without data
        messages.append({"role": "user", "content": question})
    
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    
    if not resp.choices:
        raise RuntimeError(f"No response: {resp}")
    
    return resp.choices[0].message.content.strip()
