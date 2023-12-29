# Medical Q&A ChatBot

This repository contains the code for a Medical Q&A Chatbot powered by the RAG (Retrieval-Augmented Generation) framework. The chatbot is designed to provide accurate and informative answers by combining information retrieval from medical PDFs with language model-based text generation.

## Table of Contents
- [Dependencies](#dependencies)
- [Vector Database Creation](#description)
- [Information Retrieval and Embeddings](#setup-and-dependencies)
- [Custom Prompt Template](#project-components)
- [User-Friendly Chat Interface](#project-components)
- [RetrievalQA Chain](#project-components)
- [Model Loading and Configuration](#project-components)
- [Chatbot Initialization and Operation](#project-components)
- [Output Functionality](#project-components)

## Dependencies

- LangChain
- Chainlit
- Hugging Face Transformers
- FAISS

## Features

1. ### Vector Database Creation:
   - The code includes functionality to create a searchable vector database from medical PDFs located in the specified data directory.

2. ### Information Retrieval and Embeddings:
   - Utilizes LangChain for document loading, text splitting, and Hugging Face embeddings for efficient information retrieval.
   - Implements the FAISS vector store to organize information and enable efficient search operations.

3. ### Custom Prompt Template:
   - Defines a custom prompt template to guide the language model in generating medically accurate and helpful answers within the RAG framework.

4. ### User-Friendly Chat Interface:
   - Implements a user-friendly chat interface using Chainlit, allowing users to interact effortlessly with the RAG-based chatbot.

5. ### RetrievalQA Chain:
   - Constructs a LangChain RetrievalQA chain to seamlessly link information retrieval and language model-based text generation.
   - The chain is set up to retrieve information from the vector database and generate contextually relevant responses.

6. ### Model Loading and Configuration:
   - Loads the Llama 2 language model for text generation and configures it with specific parameters such as model type, maximum new tokens, and temperature.

7. ### Chatbot Initialization and Operation:
   - Initializes the chatbot by loading the language model, creating the vector database, and setting up the RetrievalQA chain.
   - Provides an asynchronous chat interface using Chainlit to start and handle user queries.

8. ### Output Functionality:
   - Defines an output function that takes a user query, runs it through the chatbot, and returns the response.

