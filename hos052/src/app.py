"""
AgroGuard AI - Plant Disease Detector and Farming Assistant

Tabs:
  1. Diagnose        - upload a leaf photo, get the disease and a care report
  2. Disease Library - browse information about every supported disease
  3. Farming Chatbot - ask simple questions about diseases

How to run:
    python src/app.py
"""

import gradio as gr

from classifier import CLASS_NAMES, MODEL_CHOICES, get_classifier
from explain import explain
from knowledge_base import chat_answer, diagnosis_report, pretty

TOP_K = 5  # Number of top predictions shown to the user

# Remembers the last diagnosed disease so the chatbot can answer
# follow-up questions such as "How do I prevent it?"
# (simple version: shared by all users of this running app)
last = {"label": None}


def diagnose(image, model_label):
    """Predict the disease from a leaf image and build the care report."""
    # Error handling: no image was uploaded
    if image is None:
        raise gr.Error("Please upload a leaf image first.")

    try:
        classifier = get_classifier(MODEL_CHOICES[model_label])
        predictions = classifier.predict(image, top_k=TOP_K)
    except FileNotFoundError as error:
        raise gr.Error(str(error))
    except Exception as error:
        raise gr.Error(f"Could not process the image: {error}")

    best_label, best_conf = predictions[0]
    last["label"] = best_label
    scores = {pretty(label): prob for label, prob in predictions}
    return scores, diagnosis_report(best_label, best_conf)


def show_library(label):
    """Show the information page of the selected disease."""
    return diagnosis_report(label) if label else "Please select a disease."


def chatbot_fn(message, history):
    """Answer a chatbot question (uses the last diagnosis as context)."""
    return chat_answer(message, last["label"])


def explain_image(image, model_label):
    """Create a Grad-CAM heatmap that shows where the model looked."""
    # Error handling: no image was uploaded
    if image is None:
        raise gr.Error("Please upload a leaf image first.")

    try:
        classifier = get_classifier(MODEL_CHOICES[model_label])
        overlay, class_index, probability = explain(classifier, image)
    except FileNotFoundError as error:
        raise gr.Error(str(error))
    except Exception as error:
        raise gr.Error(f"Could not create the heatmap: {error}")

    text = (
        f"**Predicted:** {pretty(CLASS_NAMES[class_index])} ({probability * 100:.0f}% confidence)\n\n"
        "Red and yellow areas are the parts of the leaf that influenced this decision the most. "
        "Blue areas had little influence. If the red area is on the background instead of the leaf, "
        "the prediction is probably not reliable."
    )
    return overlay, text


# Dropdown choices for the library: (readable name, raw label)
library_choices = sorted([(pretty(name), name) for name in CLASS_NAMES])

with gr.Blocks(title="AgroGuard") as demo:
    gr.Markdown(
        "# AgroGuard: Plant Disease Detector and Farming Assistant\n"
        "Upload a leaf photo to detect the disease, then get disease information, "
        "plant-care suggestions and a prevention checklist."
    )
    with gr.Tabs():
        with gr.Tab("Diagnose"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label="Leaf image")
                    model_dd = gr.Dropdown(
                        choices=list(MODEL_CHOICES.keys()),
                        value=list(MODEL_CHOICES.keys())[0],
                        label="Choose a model",
                    )
                    btn = gr.Button("Diagnose", variant="primary")
                with gr.Column():
                    label_out = gr.Label(num_top_classes=TOP_K, label="Most likely diseases")
            report_out = gr.Markdown()
            btn.click(diagnose, inputs=[img_in, model_dd], outputs=[label_out, report_out])

        with gr.Tab("Disease Library"):
            dd = gr.Dropdown(choices=library_choices, label="Select a disease")
            lib_btn = gr.Button("Show information", variant="primary")
            lib_out = gr.Markdown()
            lib_btn.click(show_library, inputs=dd, outputs=lib_out)
            dd.input(show_library, inputs=dd, outputs=lib_out)

        with gr.Tab("Farming Chatbot"):
            gr.ChatInterface(
                fn=chatbot_fn,
                examples=[
                    "How can I prevent Early Blight?",
                    "What are the symptoms of tomato late blight?",
                    "How does apple scab spread?",
                    "Which pesticide and dosage should I use?",
                ],
            )

        with gr.Tab("Explain (Grad-CAM)"):
            gr.Markdown(
                "**Explainable AI:** see where the model looked when it made its decision. "
                "The heatmap is created with Grad-CAM."
            )
            with gr.Row():
                with gr.Column():
                    ex_img = gr.Image(type="pil", label="Leaf image")
                    ex_model = gr.Dropdown(
                        choices=list(MODEL_CHOICES.keys()),
                        value=list(MODEL_CHOICES.keys())[0],
                        label="Choose a model",
                    )
                    ex_btn = gr.Button("Show heatmap", variant="primary")
                with gr.Column():
                    ex_out = gr.Image(label="Grad-CAM heatmap")
                    ex_text = gr.Markdown()
            ex_btn.click(explain_image, inputs=[ex_img, ex_model], outputs=[ex_out, ex_text])

    gr.Markdown("_This is a support tool. Please consult a local agricultural officer before making final decisions._")

if __name__ == "__main__":
    demo.launch()
