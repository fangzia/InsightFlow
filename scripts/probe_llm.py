"""Quick connectivity probe for the configured LLM endpoint."""

import httpx

from insightflow.core.config import get_settings


def main() -> None:
    s = get_settings()
    if not s.openai_api_key:
        raise SystemExit("INSIGHTFLOW_OPENAI_API_KEY not set")
    url = s.openai_base_url.rstrip("/") + "/chat/completions"
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {s.openai_api_key}"},
        json={
            "model": s.default_model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 512,
        },
        timeout=180,
    )
    print("status:", resp.status_code)
    print(resp.json().get("choices", [{}])[0].get("message", {}).get("content"))


if __name__ == "__main__":
    main()
