from datetime import datetime


def get_sentiment_color(sentiment):
    colors = {
        "Positive": "#16a34a",
        "Negative": "#dc2626",
        "Neutral": "#d97706",
    }
    return colors.get(sentiment, "#64748b")


def calculate_distribution(results):
    total = len(results)
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for result in results:
        sentiment = result["sentiment"]
        if sentiment in counts:
            counts[sentiment] += 1

    if total == 0:
        return {label: 0 for label in counts}
    return {
        label: round(count / total * 100, 2)
        for label, count in counts.items()
    }


def create_history_record(text, result):
    return {
        "text": text,
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
