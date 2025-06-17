# InsightZentra 

A web app that:
- Accepts a URL and summarizes article content
- Supports user login/signup (MySQL with XAMPP)
- Enables question answering from the article
- Offers multilingual translation
- Plays audio narration (TTS)
- Exports content as PDF or DOCX

## Tech Stack

- Flask (Python)
- MySQL (via XAMPP)
- PHP (for user auth - optional)
- Langchain + Groq API (summarization)
- gTTS (text-to-speech)
- HTML/CSS + JS

## Setup Instructions

1. Clone the repo:
      git clone https://github.com/SNilesh2/InsightZentra.git

2. Install Python dependencies:
     pip install -r requirements.txt
  
3. Configure `.env`:
      GROQ_API_KEY=your_groq_key

4. Start XAMPP and ensure MySQL + Apache are running.

5. Run Flask app:
   python app.py
   

## Features

- Multilingual article summarization
- Q&A with context memory
- Export to PDF & DOCX
- Audio generation for summaries & answers

##  License

MIT
