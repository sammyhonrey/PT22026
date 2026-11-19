import csv
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.src.history import add_history_record, clear_history, get_history
from backend.src.sentiment import analyze_batch, predict_sentiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
DEMO_SEED_MARKER = DATA_FOLDER / "demo_reviews_seeded.json"
MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "models"
    / "smartfeel_roberta_calibrated_final"
    / "calibration_config.json"
)

logger = logging.getLogger("smartfeel")
MODEL_LOCK = threading.Lock()
HISTORY_LOCK = threading.RLock()
SENTIMENTS = ("Positive", "Neutral", "Negative")

DEMO_REVIEWS = (
    {
        "demo_id": "review-headphones-maya-chen",
        "customer_name": "Maya Chen",
        "product_name": "UltraSound Pro ANC Headphones v3",
        "review_text": (
            "The noise cancellation is definitely top-tier and beats my older Bose "
            "pair easily on flights. Soundstage is crisp with punchy bass. However, "
            "the companion mobile app disconnects intermittently and the earcups "
            "get warm after two hours of continuous wear. Still worth the price "
            "for frequent travel."
        ),
    },
    {
        "demo_id": "review-desk-chair-jordan-lee",
        "customer_name": "Jordan Lee",
        "product_name": "Apex Ergonomic Desk Chair",
        "review_text": (
            "The lumbar support broke after two weeks, and the chair is already "
            "creaking. Assembly instructions were confusing and support still "
            "hasn't replied to my request for help."
        ),
    },
    {
        "demo_id": "review-coffee-maker-priya-patel",
        "customer_name": "Priya Patel",
        "product_name": "BrewMaster Compact Coffee Maker",
        "review_text": (
            "It makes a good cup of coffee and fits nicely on my counter. Setup "
            "was straightforward. The water reservoir is smaller than I expected, "
            "but overall it works as described."
        ),
    },
    {
        "demo_id": "review-travel-mug-ethan-brooks",
        "customer_name": "Ethan Brooks",
        "product_name": "TrailSip Insulated Travel Mug",
        "review_text": (
            "I love this mug. It keeps coffee hot all morning, never leaks in my "
            "bag, and is easy to clean. Excellent quality and well worth the price."
        ),
    },
)


def seed_demo_reviews():
    if DEMO_SEED_MARKER.exists():
        return

    with HISTORY_LOCK:
        existing_ids = {
            record.get("demo_id")
            for record in get_history()
            if record.get("demo_id")
        }

    missing_reviews = [
        review for review in DEMO_REVIEWS if review["demo_id"] not in existing_ids
    ]
    predictions = []
    try:
        with MODEL_LOCK:
            for review in missing_reviews:
                predictions.append(predict_sentiment(review["review_text"]))
    except Exception:
        logger.exception("Could not prepare the sample review records.")
        raise

    with HISTORY_LOCK:
        for review, result in zip(missing_reviews, predictions):
            add_history_record(
                "ReviewAI",
                (
                    f"Product: {review['product_name']}\n"
                    f"Customer: {review['customer_name']}\n"
                    f"Review: {review['review_text']}"
                ),
                result,
                {
                    "demo_id": review["demo_id"],
                    "is_demo": True,
                    "customer_name": review["customer_name"],
                    "product_name": review["product_name"],
                },
            )
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        DEMO_SEED_MARKER.write_text(
            json.dumps({"seeded": True}, indent=2),
            encoding="utf-8",
        )


@asynccontextmanager
async def lifespan(_app):
    seed_demo_reviews()
    yield


app = FastAPI(
    title="SmartFeel AI API",
    description="Sentiment analysis and customer experience API.",
    version="1.0.0",
    lifespan=lifespan,
)

for static_folder in ("css", "js", "assets"):
    static_path = FRONTEND_ROOT / static_folder
    static_path.mkdir(parents=True, exist_ok=True)
    app.mount(
        f"/{static_folder}",
        StaticFiles(directory=static_path),
        name=static_folder,
    )


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=20000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be blank.")
        return value


class ReviewRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    product_name: str = Field(min_length=1, max_length=300)
    review_text: str = Field(min_length=1, max_length=20000)
    customer_name: str = Field(default="", max_length=150)
    is_demo: bool = False

    @field_validator("product_name", "review_text")
    @classmethod
    def fields_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("This field cannot be blank.")
        return value


class SocialRequest(BaseModel):
    comments: list[str] = Field(min_length=2, max_length=30)
    is_demo: bool = False

    @field_validator("comments")
    @classmethod
    def comments_must_contain_text(cls, values):
        cleaned = [comment.strip() for comment in values]
        if any(not comment for comment in cleaned):
            raise ValueError("Comments cannot be blank.")
        return cleaned


def serialize_prediction(result):
    scores = result["scores"]
    return {
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "scores": scores,
        "probabilities": {
            "negative": scores["Negative"],
            "neutral": scores["Neutral"],
            "positive": scores["Positive"],
        },
    }


def run_prediction(text, source, saved_text=None, metadata=None):
    try:
        with MODEL_LOCK:
            result = predict_sentiment(text)
            with HISTORY_LOCK:
                add_history_record(source, saved_text or text, result, metadata)
        return serialize_prediction(result)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Sentiment analysis failed.")
        raise HTTPException(
            status_code=500,
            detail="Sentiment analysis failed. Please try again.",
        ) from error


def filtered_history(search="", sentiment="All", source="All", days=None):
    with HISTORY_LOCK:
        records = get_history()

    cutoff = None
    if days is not None:
        cutoff = datetime.now().timestamp() - days * 24 * 60 * 60

    filtered = []
    normalized_search = search.casefold().strip()
    for record in records:
        if sentiment != "All" and record.get("sentiment") != sentiment:
            continue
        if source != "All" and record.get("source") != source:
            continue
        record_text = record.get("text")
        if normalized_search and (
            not isinstance(record_text, str)
            or normalized_search not in record_text.casefold()
        ):
            continue
        if cutoff is not None:
            try:
                record_time = datetime.strptime(
                    record.get("time", ""), "%Y-%m-%d %H:%M:%S"
                )
            except (TypeError, ValueError):
                continue
            if record_time.timestamp() < cutoff:
                continue
        filtered.append(record)
    return filtered


@app.get("/", include_in_schema=False)
def serve_frontend():
    index_path = FRONTEND_ROOT / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend files are not installed.")
    return FileResponse(index_path)


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):
    return run_prediction(request.text, "AI Playground")


@app.post("/api/review")
def analyze_review(request: ReviewRequest):
    customer = request.customer_name.strip()
    saved_text = f"Product: {request.product_name}\n"
    if customer:
        saved_text += f"Customer: {customer}\n"
    saved_text += f"Review: {request.review_text}"
    result = run_prediction(
        request.review_text,
        "ReviewAI",
        saved_text,
        {
            "product_name": request.product_name,
            "customer_name": customer,
            "is_demo": request.is_demo,
        },
    )
    return {
        **result,
        "product_name": request.product_name,
        "customer_name": customer,
        "review_text": request.review_text,
        "is_demo": request.is_demo,
    }


@app.post("/api/social")
def analyze_social(request: SocialRequest):
    try:
        with MODEL_LOCK:
            results = analyze_batch(request.comments)
            with HISTORY_LOCK:
                for result in results:
                    metadata = {"is_demo": True} if request.is_demo else None
                    add_history_record("SocialMind", result["text"], result, metadata)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Social analysis failed.")
        raise HTTPException(
            status_code=500,
            detail="Social analysis failed. Please try again.",
        ) from error

    counts = Counter(result["sentiment"] for result in results)
    total = len(results)
    overall = max(SENTIMENTS, key=lambda label: counts[label])
    distribution = {
        label: {
            "count": counts[label],
            "percentage": round(counts[label] / total * 100, 2),
        }
        for label in SENTIMENTS
    }
    return {
        "results": [
            serialize_prediction(result)
            | {"text": result["text"], "is_demo": request.is_demo}
            for result in results
        ],
        "total_comments_analyzed": total,
        "overall_sentiment": overall,
        "distribution": distribution,
    }


@app.get("/api/history")
def history(
    search: str = Query(default="", max_length=500),
    sentiment: Literal["All", "Positive", "Neutral", "Negative"] = "All",
    source: str = Query(default="All", max_length=100),
    days: int | None = Query(default=None, ge=1, le=3650),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    records = filtered_history(search, sentiment, source, days)
    return {
        "records": records[offset : offset + limit],
        "total": len(records),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/history/export.csv")
def export_history(
    search: str = Query(default="", max_length=500),
    sentiment: Literal["All", "Positive", "Neutral", "Negative"] = "All",
    source: str = Query(default="All", max_length=100),
    days: int | None = Query(default=None, ge=1, le=3650),
):
    records = filtered_history(search, sentiment, source, days)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "Source",
            "Text",
            "Sentiment",
            "Confidence",
            "Time",
            "Customer",
            "Product",
            "Demo",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record.get("source", ""),
                record.get("text", ""),
                record.get("sentiment", ""),
                record.get("confidence", ""),
                record.get("time", ""),
                record.get("customer_name", ""),
                record.get("product_name", ""),
                "Yes" if record.get("is_demo") else "No",
            ]
        )
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="smartfeel_history.csv"'},
    )


@app.delete("/api/history")
def delete_history():
    with HISTORY_LOCK:
        clear_history()
    return {"cleared": True}


@app.get("/api/dashboard")
def dashboard(
    days: int | None = Query(default=None, ge=1, le=3650),
    source: str = Query(default="All", max_length=100),
):
    all_records = filtered_history(source=source)
    previous_records = []
    if days is None:
        records = all_records
    else:
        now = datetime.now().timestamp()
        cutoff = now - days * 24 * 60 * 60
        previous_cutoff = cutoff - days * 24 * 60 * 60
        records = []
        for record in all_records:
            try:
                record_time = datetime.strptime(
                    record.get("time", ""), "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            except (TypeError, ValueError):
                continue
            if record_time >= cutoff:
                records.append(record)
            elif previous_cutoff <= record_time < cutoff:
                previous_records.append(record)

    counts = Counter(
        record.get("sentiment")
        for record in records
        if record.get("sentiment") in SENTIMENTS
    )
    previous_counts = Counter(
        record.get("sentiment")
        for record in previous_records
        if record.get("sentiment") in SENTIMENTS
    )

    def percent_change(current, previous):
        return round((current - previous) / previous * 100, 1) if previous else None

    period_change = None
    if days is not None:
        period_change = {
            "total": percent_change(len(records), len(previous_records)),
            "distribution": {
                label: percent_change(counts[label], previous_counts[label])
                for label in SENTIMENTS
            },
        }

    total = len(records)
    confidence_values = [
        float(record["confidence"])
        for record in records
        if isinstance(record.get("confidence"), (int, float))
        and not isinstance(record.get("confidence"), bool)
    ]
    confidence_by_label = {label: [] for label in SENTIMENTS}
    for record in records:
        label = record.get("sentiment")
        confidence = record.get("confidence")
        if (
            label in SENTIMENTS
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
        ):
            confidence_by_label[label].append(float(confidence))

    distribution = {
        label: {
            "count": counts[label],
            "percentage": round(counts[label] / total * 100, 2) if total else 0,
            "mean_confidence": (
                round(sum(confidence_by_label[label]) / len(confidence_by_label[label]), 2)
                if confidence_by_label[label]
                else None
            ),
        }
        for label in SENTIMENTS
    }

    grouped_activity = {}
    for record in records:
        try:
            record_date = datetime.strptime(
                record.get("time", ""), "%Y-%m-%d %H:%M:%S"
            ).date()
        except (TypeError, ValueError):
            continue
        bucket = grouped_activity.setdefault(
            record_date,
            {"date": record_date.isoformat(), "Positive": 0, "Neutral": 0, "Negative": 0},
        )
        sentiment = record.get("sentiment")
        if sentiment in SENTIMENTS:
            bucket[sentiment] += 1

    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    return {
        "total_analyses": total,
        "period_change": period_change,
        "mean_confidence": (
            round(sum(confidence_values) / len(confidence_values), 2)
            if confidence_values
            else None
        ),
        "counts": {label: counts[label] for label in SENTIMENTS},
        "distribution": distribution,
        "activity": [
            grouped_activity[activity_date]
            for activity_date in sorted(grouped_activity)
        ],
        "recent_analyses": records[:8],
        "demo_records_count": sum(1 for record in records if record.get("is_demo")),
        "model_evaluation": {
            "accuracy": float(model_config["test_accuracy"]) * 100,
            "macro_f1": float(model_config["test_macro_f1"]),
            "dataset": "CardiffNLP TweetEval",
            "base_model": model_config["base_model"],
            "model_name": model_config["model_name"],
            "is_offline_test_metric": True,
        },
    }
