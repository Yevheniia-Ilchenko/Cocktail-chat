from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.schema import Document
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Dict
import re

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

path = "cocktails/final_cocktails.csv"
cocktails_df = pd.read_csv(path)
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
user_memory = {"favourite_ingredients": []}

cocktails_docs = cocktails_df.apply(
    lambda row: Document(
        page_content=f"""
        name: {row['name']}
        ingredients: {row['ingredients']}
        alcoholic: {row['alcoholic']}
        category: {row['category']}
        glassType: {row['glassType']}
        instructions: {row['instructions']}
        """,
        metadata={
            "name": row["name"],
            "alcoholic": row["alcoholic"],
            "category": row["category"],
            "ingredients": row["ingredients"],
            "glassType": row["glassType"],
            "instructions": row["instructions"]
        }
    ),
    axis=1
).tolist()


vectorstore = FAISS.from_documents(cocktails_docs, embeddings)

SYSTEM_PROMPT = """
You are an expert cocktail recommendation system. Your responsibilities:

1. CONTEXT ANALYSIS
- Use only the information provided in the context
- Extract cocktail details: names, ingredients, types, and instructions
- Track user's favorite ingredients for personalized recommendations

2. RESPONSE FORMATTING
- Always respond in English, even if the context or question is in another language.
- Keep answers concise but informative
- For cocktail recommendations, include:
  * Cocktail names
  * Key ingredients
  * Brief preparation notes if relevant

3. MEMORY MANAGEMENT
When user mentions their favorite ingredients:
- Parse and store this information
- Use it for future personalized recommendations
- Include previously stored favorites in recommendations

4. QUERY TYPES TO HANDLE
- Finding cocktails with specific ingredients
- Recommending similar cocktails
- Recommending based on favorite ingredients
- Listing alcoholic/non-alcoholic options
- General cocktail information

5. RAG UTILIZATION
- Search for relevant cocktails in the provided database
- Compare ingredients for similarity matching
- Consider user preferences from memory

Available Context:
{context}

User Query: {question}

Respond in English, providing clear and specific recommendations based on the context and user preferences.
"""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
chain = create_stuff_documents_chain(llm=llm,
    prompt=prompt,
    document_variable_name="context")

async def generate_response(query: str) -> str:

    if any(phrase in query.lower() for phrase in ['favorite', 'favourite', 'i like']):
        update_user_memory(query)
    
    relevant_docs = vectorstore.similarity_search(query, k=5)
    
    memory_context = ""
    if user_memory["favourite_ingredients"]:
        memory_context = f"\nfavorite ingredients users: {', '.join(user_memory['favourite_ingredients'])}"
    
    docs = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in relevant_docs
    ]
    
    if memory_context:
        docs.append(Document(page_content=memory_context))
    
    try:
        response = await chain.ainvoke({
            "context": docs,
            "question": query
        })
        
        if isinstance(response, dict) and 'answer' in response:
            return response['answer']
        else:
            return str(response)
            
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        return "Error in generate_response"

def update_user_memory(query: str) -> None:
    query = query.lower()
    patterns = [
        r'(?:favorite|favourite) ingredients? (?:is|are)\s*([\w\s,and]+)',
        r'(?:my|i) (?:favorite|favourite) ingredient is\s*([\w\s]+)', 
        r'i like\s*([\w\s]+)',                                             
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            ingredients_text = match.group(1)
            ingredients_text = ingredients_text.replace(' and ', ', ')
            ingredients = [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
            
            for ingredient in ingredients:
                if ingredient not in user_memory["favourite_ingredients"]:
                    user_memory["favourite_ingredients"].append(ingredient)
            break

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/chat")
async def chat_get(query: str):
    response = await generate_response(query)
    return {"response": response}

@app.post("/chat")
async def chat_post(request: QueryRequest):
    response = await generate_response(request.query)
    return {"response": response}

@app.get("/favourite-ingredients")
async def favourite_ingredients():
    return {"favourite_ingredients": user_memory["favourite_ingredients"]}
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)