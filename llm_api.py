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
    model: str = "gpt-4",
) -> str:
    """
    Parse tickers, Category, and Secondary Category out of `question`
    and only send those rows to the model. If none are found, falls
    back to the first `max_rows` of df.
    """
    # 1) Gather all valid values from df
    all_tickers    = set(df["Ticker"].astype(str))
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
        df2 = df2[df2["Ticker"].isin(tickers)]
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
        {"role": "system", "content": "You are a data‑savvy assistant for Canadian ETFs."},
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
