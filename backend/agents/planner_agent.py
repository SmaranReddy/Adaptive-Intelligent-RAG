from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

class PlannerAgent:
    def plan(self, user_input):
        prompt = (
            f"Extract the most relevant keywords from the research query.\n"
            f"Return **only a comma-separated list** of keywords, nothing else.\n\n"
            f"Query: {user_input}"
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        keywords = response.choices[0].message.content.strip()

        # Ensure only keywords are returned
        if ':' in keywords:
            keywords = keywords.split(':')[-1].strip()

        print(f"🧩 Extracted Keywords: {keywords}")
        return keywords

