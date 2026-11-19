import { analyzeReview } from "./api.js";
import { renderPrediction, setBusy, setPlaceholder, showError } from "./ui.js";

const sampleReview = {
  product: "UltraSound Pro ANC Headphones v3",
  customer: "Maya Chen",
  review: "The noise cancellation is definitely top-tier and beats my older Bose pair easily on flights. Soundstage is crisp with punchy bass. However, the companion mobile app disconnects intermittently and the earcups get warm after two hours of continuous wear. Still worth the price for frequent travel.",
};

export function initReview() {
  const product = document.querySelector("#review-product");
  const customer = document.querySelector("#review-customer");
  const review = document.querySelector("#review-text");
  const button = document.querySelector("#review-analyze");
  const output = document.querySelector("#review-result");

  document.querySelector("#review-load-sample").addEventListener("click", () => {
    product.value = sampleReview.product;
    customer.value = sampleReview.customer;
    review.value = sampleReview.review;
    document.querySelector("#review-is-demo").checked = true;
    showError("review-error");
    product.focus();
  });

  button.addEventListener("click", async () => {
    showError("review-error");
    if (!product.value.trim()) {
      showError("review-error", "Enter the product or service name.");
      product.focus();
      return;
    }
    if (!review.value.trim()) {
      showError("review-error", "Enter a customer review.");
      review.focus();
      return;
    }

    setBusy(button, true, "Analyzing review…");
    try {
      const prediction = await analyzeReview(
        product.value.trim(),
        review.value.trim(),
        customer.value.trim(),
        document.querySelector("#review-is-demo").checked,
      );
      renderPrediction(
        output,
        prediction,
        prediction.product_name,
        prediction.customer_name,
        prediction.is_demo,
      );
    } catch (error) {
      showError("review-error", error.message);
    } finally {
      setBusy(button, false);
    }
  });

  document.querySelector("#review-clear").addEventListener("click", () => {
    product.value = "";
    customer.value = "";
    review.value = "";
    document.querySelector("#review-is-demo").checked = false;
    showError("review-error");
    setPlaceholder(output, "Product Sentiment Diagnostics", "Submit a product review to view the model's classification.", "▧");
    product.focus();
  });
}
