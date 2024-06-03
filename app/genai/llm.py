from llama_index.llms.gemini import Gemini
from langchain_google_genai import GoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from credential import gemini_api

safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

def get_llamaindex_llm():
    llm = Gemini(api_key=gemini_api, 
                model_name='models/gemini-pro', 
                temperature=0,
                safety_settings=safety_settings
                )
    return llm

def get_langchain_llm():
    llm = GoogleGenerativeAI(
            model="gemini-pro", 
            google_api_key=gemini_api,
            safety_settings=safety_settings
            )
    return llm

def get_gemini():
    genai.configure(api_key=gemini_api)
    llm = genai.GenerativeModel(
        'gemini-pro', 
        safety_settings=safety_settings
        )
    return llm
