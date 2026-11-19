import gradio as gr
import joblib

# Load trained model
model = joblib.load("models/spam_classifier.joblib")


# Email prediction function
def predict_email(subject, message):

    if not subject.strip() and not message.strip():
        return "Please enter an email subject or message."

    email_text = subject + " " + message

    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]

    confidence = max(probabilities) * 100

    if prediction == "spam":
        return (
            f"🚨 SPAM EMAIL DETECTED\n\n"
            f"Confidence: {confidence:.2f}%\n\n"
            f"This email may be suspicious."
        )

    else:
        return (
            f"✅ NOT SPAM (HAM)\n\n"
            f"Confidence: {confidence:.2f}%\n\n"
            f"This email is classified as normal."
        )


# Custom CSS
custom_css = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
    color: #174A7E;
}

#subtitle {
    text-align: center;
    color: #555555;
}
"""


# Create Gradio Interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css,
    title="Spam Email Detection System"
) as demo:

    gr.Markdown(
        "# Spam Email Detection System",
        elem_id="main-title"
    )

    gr.Markdown(
        "AI-powered email classification using Machine Learning.",
        elem_id="subtitle"
    )

    with gr.Row():

        with gr.Column():

            subject = gr.Textbox(
                label="Email Subject",
                placeholder="Enter email subject...",
                lines=2
            )

            message = gr.Textbox(
                label="Email Body",
                placeholder="Paste your email message here...",
                lines=10
            )

            with gr.Row():

                clear_btn = gr.ClearButton(
                    [subject, message],
                    value="Clear"
                )

                analyze_btn = gr.Button(
                    "Analyze Email",
                    variant="primary"
                )

        with gr.Column():

            result = gr.Textbox(
                label="Detection Result",
                lines=8,
                interactive=False
            )

            gr.Markdown(
                """
                ### Privacy Notice

                This application classifies email text
                using a trained machine learning model.

                A prediction does not guarantee that
                an email is safe.
                """
            )

    analyze_btn.click(
        fn=predict_email,
        inputs=[subject, message],
        outputs=result
    )

    gr.Markdown(
        "---\n"
        "Developed using Python, Scikit-learn and Gradio."
    )


if __name__ == "__main__":
    demo.launch()