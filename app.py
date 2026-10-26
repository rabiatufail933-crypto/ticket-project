import json
import random
import re
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Intelligent Support System", page_icon="🎫", layout="wide")

# ---------------------------------------------------------------
# CONFIGURATION: categories, keywords (English + Roman Urdu + Urdu), routing
# ---------------------------------------------------------------
CATEGORIES = {
    "Billing and Payment Issues": {
        "route": "Finance Department",
        "keywords": ["pay", "payment", "charge", "charged", "bill", "invoice", "money", "price",
                     "deducted", "card", "paise", "paisa", "kat gaye", "kat gaya",
                     "پیسے", "ادائیگی", "بل", "چارج"],
    },
    "Shipping and Delivery Issues": {
        "route": "Logistics Department",
        "keywords": ["ship", "deliver", "delivery", "track", "order", "courier", "address",
                     "parcel", "late", "delay", "kab milega", "nahi mila", "nahi aaya",
                     "آرڈر", "ڈیلیوری", "پارسل", "نہیں ملا"],
    },
    "Refunds and Returns": {
        "route": "Accounts / Returns Team",
        "keywords": ["refund", "return", "cancel", "exchange", "wapas", "waapis", "wapis",
                     "واپس", "ریفنڈ", "منسوخ"],
    },
    "Customer Feedback and Suggestions": {
        "route": "Product Management",
        "keywords": ["suggest", "suggestion", "feedback", "improve", "idea", "review",
                     "acha laga", "behtar", "مشورہ", "رائے", "بہتر"],
    },
    "Support Issues": {
        "route": "Technical Support",
        "keywords": ["login", "password", "error", "bug", "crash", "not working", "problem",
                     "issue", "account", "kaam nahi", "masla", "مسئلہ", "کام نہیں"],
    },
}
DEFAULT_CATEGORY = "Support Issues"

NEGATIVE = ["bad", "worst", "terrible", "angry", "fake", "fraud", "scam", "horrible", "useless",
            "disappointed", "bura", "bakwas", "ghalat", "dhoka", "bekaar", "برا", "بکواس", "دھوکہ"]
POSITIVE = ["thanks", "thank you", "good", "great", "nice", "love", "happy", "excellent",
            "shukriya", "acha", "bohat acha", "mazay ka", "شکریہ", "اچھا", "زبردست"]
URGENT = ["urgent", "immediately", "asap", "twice", "fraud", "scam", "emergency",
          "jaldi", "foran", "abhi", "فوری", "جلدی"]

ROMAN_URDU_HINTS = ["kya", "hai", "hain", "mera", "meri", "mujhe", "nahi", "nahin", "paisa", "paise",
                    "wapas", "kab", "aap", "karo", "kar", "bhej", "mila", "ka", "ki", "ko", "se", "bohat"]

AUTO_REPLY = {
    "English": "Thank you for contacting us. Your ticket has been forwarded to {route}. We will get back to you shortly.",
    "Roman Urdu": "Humse rabta karne ka shukriya. Aapka ticket {route} ko bhej diya gaya hai. Hum jaldi aap se rabta karenge.",
    "Urdu": "ہم سے رابطہ کرنے کا شکریہ۔ آپ کا ٹکٹ {route} کو بھیج دیا گیا ہے۔ ہم جلد آپ سے رابطہ کریں گے۔",
}

SENTIMENT_LABEL = {"Negative": "Negative 😡", "Neutral": "Neutral 😐", "Positive": "Positive 😊"}
PRIORITY_LABEL = {"High": "🔴 High Priority", "Medium": "🟡 Medium Priority", "Low": "🟢 Low Priority"}


# ---------------------------------------------------------------
# ANALYSIS: rule-based engine (works offline) + optional Claude AI engine
# ---------------------------------------------------------------
def detect_language(text):
    if re.search(r"[\u0600-\u06FF]", text):
        return "Urdu"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for w in words if w in ROMAN_URDU_HINTS)
    if hits >= 2:
        return "Roman Urdu"
    return "English"


def count_hits(text_lower, words):
    return sum(1 for w in words if w in text_lower)


def analyze_rules(text):
    t = text.lower()

    # Category: highest keyword score wins
    scores = {name: count_hits(t, cfg["keywords"]) for name, cfg in CATEGORIES.items()}
    best = max(scores, key=scores.get)
    category = best if scores[best] > 0 else DEFAULT_CATEGORY

    # Sentiment
    neg, pos = count_hits(t, NEGATIVE), count_hits(t, POSITIVE)
    sentiment = "Negative" if neg > pos else "Positive" if pos > neg else "Neutral"

    # Priority
    if count_hits(t, URGENT) > 0 or sentiment == "Negative":
        priority = "High"
    elif sentiment == "Positive":
        priority = "Low"
    else:
        priority = "Medium"

    return {"category": category, "sentiment": sentiment, "priority": priority,
            "language": detect_language(text), "engine": "Rule-based NLP"}


def get_provider():
    """Returns 'gemini' (free key), 'claude', or None depending on which secret is set."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return "gemini"
        if "ANTHROPIC_API_KEY" in st.secrets:
            return "claude"
    except Exception:
        pass
    return None


def ai_available():
    return get_provider() is not None


def analyze_ai(text):
    """AI-as-a-service: calls an external AI API (Gemini free tier or Claude)."""
    provider = get_provider()
    prompt = (
        "Classify this customer support ticket. Reply with JSON only, no other text, with keys: "
        f'"category" (one of {list(CATEGORIES)}), "sentiment" (Positive/Neutral/Negative), '
        '"priority" (High/Medium/Low), "language" (English/Roman Urdu/Urdu).\n\n'
        f"Ticket: {text}"
    )

    if provider == "gemini":
        import time

        import requests

        models = [st.secrets.get("GEMINI_MODEL", "gemini-flash-latest"),
                  "gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        raw, last_err = None, None
        for model in dict.fromkeys(models):  # unique, keeps order
            for attempt in range(2):
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    resp = requests.post(
                        url,
                        headers={"x-goog-api-key": st.secrets["GEMINI_API_KEY"],
                                 "Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=30,
                    )
                    if resp.status_code in (429, 500, 503):  # busy: wait and retry
                        last_err = f"{resp.status_code} on {model}"
                        time.sleep(1.5)
                        continue
                    resp.raise_for_status()
                    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    break
                except Exception as e:
                    last_err = str(e)
                    break
            if raw:
                break
        if not raw:
            raise RuntimeError(f"Gemini unavailable ({last_err})")
        engine = "Gemini AI (API)"
    else:
        import anthropic

        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        engine = "Claude AI (API)"

    data = json.loads(re.search(r"\{.*\}", raw, re.S).group())
    if data.get("category") not in CATEGORIES:
        data["category"] = DEFAULT_CATEGORY
    data["engine"] = engine
    return data


def analyze_ticket(text, use_ai):
    if use_ai:
        try:
            result = analyze_ai(text)
        except Exception as e:
            st.warning(f"AI engine failed, using rule-based fallback. ({e})")
            result = analyze_rules(text)
    else:
        result = analyze_rules(text)
    result["route"] = CATEGORIES[result["category"]]["route"]
    return result


# ---------------------------------------------------------------
# DATA: demo history so dashboard is not empty
# ---------------------------------------------------------------
def make_demo_data(n=124):
    rnd = random.Random(42)
    cats = list(CATEGORIES)
    rows = []
    for i in range(n):
        cat = rnd.choices(cats, weights=[25, 25, 20, 10, 20])[0]
        sent = rnd.choices(["Negative", "Neutral", "Positive"], weights=[25, 50, 25])[0]
        pri = "High" if (sent == "Negative" and rnd.random() < 0.55) else ("Low" if sent == "Positive" else "Medium")
        rows.append({
            "Time": f"2026-10-{rnd.randint(1, 25):02d}",
            "Ticket": f"Demo ticket #{i + 1}",
            "Category": cat,
            "Sentiment": sent,
            "Priority": pri,
            "Language": rnd.choices(["English", "Roman Urdu", "Urdu"], weights=[60, 30, 10])[0],
            "Route": CATEGORIES[cat]["route"],
            "Engine": "Demo data",
        })
    return pd.DataFrame(rows)


if "tickets" not in st.session_state:
    st.session_state.tickets = make_demo_data()

df = st.session_state.tickets

# ---------------------------------------------------------------
# SIDEBAR: live admin metrics
# ---------------------------------------------------------------
st.sidebar.header("📊 Admin Dashboard & Metrics")
st.sidebar.metric("Total Tickets Processed", len(df))
st.sidebar.metric("High Priority Tickets", int((df["Priority"] == "High").sum()))
st.sidebar.metric("Avg Resolution Time", "2.4 Hours")

st.sidebar.subheader("Team Routing Status")
for team in sorted({cfg["route"] for cfg in CATEGORIES.values()}):
    n = int((df["Route"] == team).sum())
    st.sidebar.text(f"🟢 {team}: Active ({n})")

st.sidebar.subheader("⚙️ Analysis Engine")
use_ai = st.sidebar.toggle("Use AI (API)", value=False, disabled=not ai_available())
if not ai_available():
    st.sidebar.caption("AI mode off: add GEMINI_API_KEY (free) in Streamlit secrets to enable.")
else:
    st.sidebar.caption(f"AI provider ready: {get_provider().title()}")

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
st.title("🎫 Intelligent Customer Support System (SaaS)")
st.markdown("Automated Ticket Classification, Sentiment Analysis, Multilingual Support & Smart Routing")

tab1, tab2, tab3 = st.tabs(["📝 Submit Ticket", "📈 Dashboard", "🗂️ Ticket History"])

with tab1:
    st.subheader("Submit or Test Customer Ticket")
    examples = {
        "Custom": "",
        "Billing (English)": "My payment was deducted twice for the invoice, please refund my money immediately!",
        "Shipping (Roman Urdu)": "Mera order abhi tak nahi mila, kab milega? Please tracking bhejein.",
        "Refund (Urdu)": "میرا آرڈر واپس کرنا ہے، مجھے ریفنڈ چاہیے",
        "Feedback (English)": "Great app, thanks! I have a suggestion to improve the search.",
        "Support (English)": "I cannot login, the app shows an error and keeps crashing.",
    }
    choice = st.selectbox("Load an example ticket (optional):", list(examples))
    user_input = st.text_area("Enter Customer Ticket / Query:", value=examples[choice], height=120)

    if st.button("Analyze & Route Ticket", type="primary"):
        if user_input.strip():
            r = analyze_ticket(user_input, use_ai)
            st.success(f"Analysis Complete! (Engine: {r['engine']})")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.info("Category"); c1.write(r["category"])
            c2.info("Sentiment"); c2.write(SENTIMENT_LABEL.get(r["sentiment"], r["sentiment"]))
            c3.info("Priority Level"); c3.write(PRIORITY_LABEL.get(r["priority"], r["priority"]))
            c4.info("Language"); c4.write(r["language"])
            c5.info("Automated Route"); c5.write(f"➡️ {r['route']}")

            if r["priority"] == "High":
                st.error("🚩 This ticket is flagged as HIGH PRIORITY and escalated.")

            st.markdown("**Suggested auto-reply:**")
            st.code(AUTO_REPLY.get(r["language"], AUTO_REPLY["English"]).format(route=r["route"]), language=None)

            new_row = {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Ticket": user_input.strip()[:80],
                "Category": r["category"],
                "Sentiment": r["sentiment"],
                "Priority": r["priority"],
                "Language": r["language"],
                "Route": r["route"],
                "Engine": r["engine"],
            }
            st.session_state.tickets = pd.concat(
                [st.session_state.tickets, pd.DataFrame([new_row])], ignore_index=True
            )
            st.caption("Ticket saved. Open the Dashboard / History tabs to see it.")
        else:
            st.warning("Please enter a ticket text first.")

with tab2:
    st.subheader("Interactive Dashboard")
    f1, f2 = st.columns(2)
    pri_filter = f1.multiselect("Filter by priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    lang_filter = f2.multiselect("Filter by language", ["English", "Roman Urdu", "Urdu"],
                                 default=["English", "Roman Urdu", "Urdu"])
    view = st.session_state.tickets
    view = view[view["Priority"].isin(pri_filter) & view["Language"].isin(lang_filter)]

    k1, k2, k3 = st.columns(3)
    k1.metric("Tickets (filtered)", len(view))
    k2.metric("High Priority", int((view["Priority"] == "High").sum()))
    k3.metric("Negative Sentiment", int((view["Sentiment"] == "Negative").sum()))

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Tickets by Category**")
        st.bar_chart(view["Category"].value_counts())
    with g2:
        st.markdown("**Sentiment Distribution**")
        st.bar_chart(view["Sentiment"].value_counts())
    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**Priority Levels**")
        st.bar_chart(view["Priority"].value_counts())
    with g4:
        st.markdown("**Tickets per Team (Routing)**")
        st.bar_chart(view["Route"].value_counts())

with tab3:
    st.subheader("Ticket History")
    st.dataframe(st.session_state.tickets.iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download as CSV",
        st.session_state.tickets.to_csv(index=False).encode("utf-8"),
        "tickets.csv",
        "text/csv",
    )
