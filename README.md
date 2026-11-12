# Rhetoric Lab — Free-tier LLM Streamlit Chatbot

Interactive multi-turn ChatGPT-style prototype for **Neutral**, **Diplomatic**, **Trump-inspired** and **Academic** voices. Switch style during a conversation without losing your earlier turns. Threads live in Streamlit session state (not permanent storage).

## Start

Python 3.10+ required.

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Free API option A: Gemini (recommended)

1. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey).
2. In the Streamlit sidebar, choose **Gemini (free tier)**.
3. Paste the key in the password field; the default model is `gemini-2.5-flash`.
4. Turn off **Offline UI demo (scripted)** and begin chatting.

Google's free tier is subject to changing model availability, usage quotas and data-use rules. Check the current Google AI Studio project quota.

### Free API option B: Groq

1. Create a key at [GroqCloud](https://console.groq.com/keys).
2. Choose **Groq (free plan)**. The default model is `qwen/qwen3.8-27b` (available in Groq documentation as of November 2026). `llama-3.3-70b-versatile` is no longer available on Groq free/developer plans since August 16, 2026.
3. Paste the key and disable the offline demo.

### Free API option C: OpenRouter

1. Create a key at [OpenRouter](https://openrouter.ai/settings/keys).
2. Choose **OpenRouter (free models)**, default model `openrouter/free`.
3. Paste the key and disable the offline demo.

`openrouter/free` chooses an eligible available free model; since this can change between requests, outputs may be less consistent for research. For controlled comparisons, pin a specific listed free model ID instead. Free models have limits (commonly 50 requests/day for an account without credits).

### Optional: environment variables instead of sidebar entry

Set one of `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY`. Do not commit API keys to repositories.

## Conversational approach

- The normal live mode sends **one LLM request per user turn**, including the full earlier conversation and the selected style's system instruction. This is a better default for fluid conversation and free-tier quotas.
- Toggle **Research mode: baseline → rewrite (2 calls)** for controlled style transformations. This makes two calls per turn (one call for Neutral) and may consume quota faster.
- Changing voice affects only future responses; earlier messages stay visible as written.
- The offline demo is intentionally scripted; it is not a model.
- Conversation history lasts for the active Streamlit session. You can export the active chat to JSON.

## Research cautions

- No factual or source verification is performed. Style instructions cannot guarantee semantic or numerical equivalence.
- "Trump-inspired" is an illustrative rhetorical style. The system does not impersonate the real politician or make claims on their behalf.
- The app is a prototype and does not measure credibility, neutrality, or authenticity without a separate user study.
- Free provider tiers can use submitted content for product improvement: use fictional examples only, **not actual interview transcripts or sensitive participant data**, unless proper consent and provider agreements are in place.
- Any key entered in a Streamlit UI is sent to the server running that UI. Run it locally for your own work; do not deploy a public app asking visitors to enter their keys without carefully designing secure key handling.
