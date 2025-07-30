# PyData Southampton - Craig West

# June 2025

[https://craig-west.netlify.app/](https://craig-west.netlify.app/)

## AI Agents in the Data Pipeline

<img src="./craig-west-pydata-southampton.png" width=700px>

## *We will go through set up together at the start of the workshop.*

Code examples will be able to use either Groq or OpenAI except a few which are for demo purposes only.

You will need an LLM API key.

Groq offers a free tier and uses the same API signature as OpenAI.

A few demos will use Anthropic's Claude API. You won't need an API key for Claude for the workshop but you will need to run it on your own machine. At this current time Claude is not available as a free tier but you will not have to pay money upfront of say $5 as with OpenAI.

Free Tier with Groq: https://console.groq.com/login

## Set Up

Create a virtual environment and install requirements:

To run LLMs, copy .env.sample to .env and add your OpenAI key or Groq API key:

- OPENAI_API_KEY=sk-proj-TQa...
- GROQ_API_KEY=gsk_ow4T...

As some libraries can take a long time to load (some parts of the Langchain ecosystem), there is a req07.txt that has all the necessary requireiments to run the code up to 07 which is used to understand the basics of AI Agents. (Excludes 03.6_sql_finder_script.py which needs sentence transformers and can take some time. `pip install sentence_transformers`)

Depending on when you set up, it may be best to use this first and load all the libraries as in requirements.txt. 
