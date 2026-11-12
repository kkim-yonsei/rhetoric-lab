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

## Confirm the API connection before chatting

1. In the left sidebar, choose a provider (e.g., **Groq (free plan)**).
2. Paste an API key from that same provider and check the **Model ID**.
3. Click **🔌 Test connection**. A short real request is sent to the selected model.
4. Status appears directly under the button:
   - 🟡 **Not tested**: key entered, but no request has succeeded yet.
   - 🟢 **Connected**: the model returned a real text response.
   - 🔴 **Connection failed**: authentication, model access, rate limit, or network error.
5. After success, turn **OFF** `Offline UI demo (scripted)` to chat with the real model.

Changing the key, model, or provider resets the displayed status to **Not tested** until you check again. The test consumes a small amount of your provider's quota. Your key isn't printed in the UI, log output, or exported chat JSON; do not share API keys in screenshots.

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


## Groq HTTP 401 diagnostic update
- Choose Groq under **LLM provider**.
- Revoke any API key that has appeared in screenshots or messages and create a new key.
- Paste the new key in the sidebar and click **Test connection**.
- The app first calls `GET /models` to check authentication **independently from the Model ID**.
- Only if authentication succeeds does it try the selected model. This distinguishes a 401 credential issue from a 404 invalid-model issue.
- The app never displays the submitted key. It reports only sanitized provider error type/code.
- If status stays 401 despite a newly generated key, verify that the application is pointing to Groq and that the expected key source is shown; then contact Groq support with error classification (not the key).
