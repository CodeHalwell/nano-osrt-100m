from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="sk-unsloth-YOUR_KEY",
)

response = client.chat.completions.create(
    model="current",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")


curl http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer sk-unsloth-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Search Python 3.13 features"}],
    "enable_tools": true,
    "enabled_tools": ["web_search", "python"],
    "stream": true
  }'