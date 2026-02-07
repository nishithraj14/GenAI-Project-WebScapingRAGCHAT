# Multi PDF Rag Chatbot built on Web Scraping


# 🚀 Live Demo – Multi-PDF Web-Scraping RAG Chatbot

👉 https://genai-project-chatscholar-4op98aa2kqklfszrgedmpe.streamlit.app/

> ⚠️ This is a deployed Streamlit demo (free tier).  
> The application may take ~15–30 seconds to wake up after inactivity.


![multipdfgif](https://github.com/user-attachments/assets/d52672eb-3a81-4922-a132-b544cf6a5dcf)


### Demo Scope
This live demo allows recruiters and users to:

- Chat with multiple PDFs using Retrieval-Augmented Generation (RAG)
- View grounded answers strictly sourced from documents
- Experience conversational memory-aware document querying
- Explore the system without any local setup

> The full production implementation (Flask + FAISS + LangChain pipeline) is available in this repository for local deployment and extensibility.

---


## Problem Statement
The growth of e-commerce platforms has led to a vast array of products and services, making it difficult for customers to find relevant information quickly. Traditional search mechanisms often fall short in delivering personalized and specific responses to customer queries. This causes frustration and longer response times, leading to potential loss in sales or customer dissatisfaction.
 Context:
Leveraging the power of machine learning, natural language & Gen AI, this tool automates the traditionally manual insurance claim processing procedure. 
-	Implementation of AI and Generative AI will enhance data analysis and   predictive capabilities.
-	AI will provide deeper insights, improve accuracy, and streamline reporting processes.
-	Predictive features will enable proactive decision-making based on anticipated impact fluctuations.

## Objective:
The goal of the ecommerce chatbot is to enhance the customer experience on the e-commerce platform by delivering fast, accurate, and personalized responses to their queries. The chatbot should:
-	Provide real-time product information and recommendations.
-	Retrieve relevant data from a knowledge base in response to customer questions.
-	Improve customer satisfaction through engaging conversations.
-	Assist in decision-making by answering product-related queries effectively.

## How it Works
The ecommerce chatbot utilizes a combination of retrieval mechanisms and generative models to deliver accurate and context-aware responses to customer queries.

## Key Components:
1.	Retrieval Mechanism:
o	The system maintains a pre-processed e-commerce knowledge base (product descriptions, FAQs, reviews, etc.).
o	When the user inputs a query, the retrieval component finds relevant documents or pieces of information related to the query from the knowledge base.
2.	Generative Model:
o	After retrieving the relevant data, the generative model (such as a GPT-based model) is tasked with crafting a human-like response using the context provided by the retrieved documents.
3.	Chatbot Pipeline:
o	Step 1: User submits a query (e.g., "What are the top features of Product X?").
o	Step 2: The retrieval component searches for relevant documents or product details related to Product X from the knowledge base.
o	Step 3: The retrieved information is passed to the generative model, which formulates a coherent, context-aware response.
o	Step 4: The response is delivered to the user in a conversational manner.

## Architecture:

![image](https://github.com/user-attachments/assets/86f014a1-548f-4ad6-8de5-8e75511e9969)
