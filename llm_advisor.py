import json
import urllib.request
import urllib.error

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_KEY = "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"
_MODEL = "openai/gpt-4o-mini"

COMPANY_LIST = [
    "NTPC", "ONGC", "SBI", "Power Grid", "Coal India", "BHEL",
    "GAIL", "IOC", "BPCL", "Bank of Baroda", "Canara Bank", "PNB",
    "HAL", "BEL", "SAIL", "NMDC", "IRCTC", "LIC",
    "Reliance", "TCS", "Infosys", "HDFC Bank", "ICICI Bank", "Wipro",
    "Adani Ports", "Bajaj Finance", "Maruti", "Tata Motors", "HUL"
]

def build_prompt(intent, predictions_summary, accuracy):
    up_count = predictions_summary["up"]
    down_count = predictions_summary["down"]
    total = up_count + down_count
    up_pct = round(up_count / total * 100) if total else 0
    avg_confidence = predictions_summary["avg_confidence"]
    avg_vol = predictions_summary["avg_vol_delta"]
    companies = intent["companies"]
    weekly_target = intent["weekly_target"]

    return f"""Generate a structured, highly analytical market advisory report based on the following model outputs and your knowledge of RECENT MARKET NEWS.

Model Output Metrics:
- Total analyzed entries: {total}
- UP bias: {up_pct}% | DOWN bias: {100 - up_pct}%
- Average confidence: {avg_confidence:.1%}
- Volatility Delta: {avg_vol:+.4f}
- Baseline model accuracy: {accuracy:.1%}

User Intent:
- Target Companies/PSUs: {companies}
- Weekly Profit Target: {weekly_target}%

IMPORTANT: YOU MUST USE YOUR INTERNAL KNOWLEDGE OF RECENT INDIAN MARKET NEWS (RBI news, earnings reports, global cues, specific PSU news) to explain these results. If the model shows an UP/DOWN bias or volatility, QUOTE specific news events that justify this movement.

IMPORTANT UI RULES: DO NOT USE HASHTAGS (#) FOR HEADERS. DO NOT USE STARS (*) FOR BOLDING. DO NOT USE MARKDOWN.
Use plain uppercase text for headers. Use simple dashes (-) for bullet points.

EXECUTIVE SUMMARY
(Write a 2-sentence summary detailing the overall bias and sentiment based on {up_pct}% UP predictions. Explicitly mention and quote 1-2 items of RECENT NEWS appearing in headlines that support this trend.)

FOCUS AREAS
(Provide 3 plain bullet points detailing specific angles the user should target, mentioning '{companies}'. Link these areas to specific current news or sector-wise updates.)

RISK ASSESSMENT
(Provide 3-4 plain bullet points outlining risk levels. Mention if specific news events like RBI announcements or global inflation data act as a threat to these predictions.)

PROJECTED GAIN/LOSS
(Provide 3 plain bullet points analyzing if the {weekly_target}% target is ACHIEVABLE. Factor in recent news-driven volatility spikes into the range estimation.)

RECOMMENDED ACTIONS
(Provide 4 tight bullet points on execution based on technical signals and news-driven sentiment.)

WHAT COULD GO WRONG
(Provide 3-4 plain bullet points of realistic events or upcoming news schedules (e.g. Fed meetings, budget updates) that could override the technical prediction.)

Keep the tone professional and crisp. Do not include conversation—just output the plain text report directly."""


def get_advisory(intent, predictions_summary, accuracy):
    prompt = build_prompt(intent, predictions_summary, accuracy)

    body = json.dumps({
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_KEY}",
            "HTTP-Referer": "https://nifty-options-ml",
            "X-Title": "NIFTY Options Advisory"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        return f"API error {e.code}: {err_body}"
    except Exception as e:
        return f"Failed to get advisory: {str(e)}"
