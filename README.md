# Career AI Assistant

A reusable bilingual AI career assistant built with Python, the OpenAI
Responses API, Gradio, PyPDF and optional Telegram notifications.

## Live production example

The private production version is available at:

https://deisy-career-ai.onrender.com

The production application uses private career documents that are not included
in this repository. This public repository contains a fictional sample profile
so the architecture can be reviewed and tested without exposing personal data.

## Features

- Answers career questions in English and Spanish.
- Grounds answers in local PDF and TXT source documents.
- Avoids inventing professional facts.
- Records questions that cannot be answered from the sources.
- Optionally sends Telegram notifications for unknown questions and contact
  requests.
- Supports local development and deployment to a web service such as Render.

## Project structure

```text
career-ai-public/
├── me/
│   └── sample_profile.txt
├── .env.example
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Local setup

1. Clone the repository.

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy `.env.example` to `.env` and add your own API key and profile values.

5. Replace `me/sample_profile.txt` with your own `.txt` or `.pdf` source files.
   Keep real documents private and do not commit them to a public repository.

6. Run the application:

```bash
python app.py
```

## Environment variables

```env
OPENAI_API_KEY=
PROFILE_NAME=
CONTACT_EMAIL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

Telegram variables are optional. If they are omitted, the assistant continues
to work without sending notifications.

## Privacy and security

- Never commit `.env`.
- Never publish API keys or Telegram tokens.
- Store production secrets in the hosting provider's secret manager.
- Use a dedicated OpenAI project with model restrictions and a spending limit.
- Keep personal source documents in a private production repository.

## Author

Deisy Tafur  
CRM & Revenue Operations | HubSpot | Marketing Automation
