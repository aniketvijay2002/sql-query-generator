# ----------------------------------------------------------
# SQL QUERY GENERATOR (LLM ONLY — NO DATABASE REQUIRED)
# Uses Groq LLaMA-3 to generate SQL queries from natural language.
# ----------------------------------------------------------

# ----------------------------------------------------------
# ADVANCED SQL QUERY + RESULT GENERATOR (LLM-ONLY)
# LangChain + Groq LLaMA
# ----------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SERVE UI ----
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# -----------------------------------------------------
# DATABASE SCHEMA
# -----------------------------------------------------
SCHEMA = """
TABLE: students
- id (INT, PRIMARY KEY)
- name (VARCHAR)
- age (INT)
- marks (INT)
- city (VARCHAR)

TABLE: courses
- id (INT, PRIMARY KEY)
- course_name (VARCHAR)
- duration (INT)

TABLE: enrollments
- student_id (INT)
- course_id (INT)
- date_enrolled (DATE)
"""

# -----------------------------------------------------
# LLM
# -----------------------------------------------------
load_dotenv()  # 👈 THIS LINE LOADS .env

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0
)


parser = StrOutputParser()

# -----------------------------------------------------
# SQL GENERATOR PROMPT
# -----------------------------------------------------
sql_prompt = ChatPromptTemplate.from_messages([
    ("system",  
     "You are an expert SQL generator.\nUse ONLY the schema below.\n\n"
     f"SCHEMA:\n{SCHEMA}"),
    ("user", "{question}")
])

sql_chain = sql_prompt | llm | parser

# -----------------------------------------------------
# RESULT SIMULATOR PROMPT
# -----------------------------------------------------
result_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You generate realistic sample output for SQL queries.\n"
     f"SCHEMA:\n{SCHEMA}"),
    ("user", "SQL Query:\n{sql_query}")
])

result_chain = result_prompt | llm | parser


class Query(BaseModel):
    question: str


@app.post("/generate")
def generate_sql(query: Query):
    sql = sql_chain.invoke({"question": query.question})
    result = result_chain.invoke({"sql_query": sql})
    return {"sql": sql, "result": result}
