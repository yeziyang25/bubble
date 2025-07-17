import pandas as pd
import json
from openai import OpenAI
from typing import List, cast
import streamlit as st



client = OpenAI(
    base_url=st.secrets["LLM_URL"], api_key=st.secrets["API_KEY"]
)

def ask_gemma(
    question: str,
    df: pd.DataFrame,
    max_rows: int = 20,
    model: str = "google/gemma-3-12b",
) -> str:
    
    sample = df.head(max_rows).copy()
    records_json = sample.to_json(orient="records", date_format="iso")
    records      = json.loads(records_json)           
    data_json    = json.dumps(records, indent=2)

    messages = [
        {"role": "system", "content": "You are a data‑savvy assistant."},
        {"role": "user",   "content": f"Data (first {max_rows} rows):\n{data_json}\n\nQ: {question}"}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,    # type: ignore
        temperature=0.0,
        max_tokens=512,
    )
    print("Full Response:", resp)  

    if not resp.choices:
       raise ValueError("The LLM returned no choices. Check the server logs for details.")

    content = resp.choices[0].message.content
    return content.strip() if content is not None else ""
