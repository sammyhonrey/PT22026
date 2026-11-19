
"""
SceneSense

Semantic, Instance and Panoptic Segmentation using MaskFormer.
"""

import gradio as gr

from segmentation.pipeline import (
    INSTANCE_MODE,
    PANOPTIC_MODE,
    SEMANTIC_MODE,
    run_segmentation,
)


# ============================================================
# CHANGE COLOURS HERE
# ============================================================
# Change these colour codes to customize the web interface.

BACKGROUND_COLOR = "#0B1020"          # Website background
PANEL_COLOR = "#141F31"               # Image panels and control boxes
FONT_COLOR = "#F8FAFC"                # Normal text and labels
HEADING_COLOR ="#F8FAFC"             # SceneSense heading

SEGMENT_BUTTON_COLOR = "#0D9488"      # Segment Image button
CLEAR_BUTTON_COLOR = "#475569"        # Clear button
BUTTON_TEXT_COLOR = "#FFFFFF"         # Text inside both buttons

SLIDER_COLOR = "#14B8A6"              # Overlay strength slider


# ============================================================
# APPLICATION THEME
# ============================================================
# Apply the selected colours to Gradio components.
# Set both normal and dark theme colours.

THEME = gr.themes.Default(
    primary_hue="teal",
    neutral_hue="slate"
).set(
    # Website background
    body_background_fill=BACKGROUND_COLOR,
    body_background_fill_dark=BACKGROUND_COLOR,

    # Image panels and control backgrounds
    block_background_fill=PANEL_COLOR,
    block_background_fill_dark=PANEL_COLOR,

    # Normal font and component label colours
    body_text_color=FONT_COLOR,
    body_text_color_dark=FONT_COLOR,
    block_label_text_color=FONT_COLOR,
    block_label_text_color_dark=FONT_COLOR,

    # Segment Image button
    button_primary_background_fill=SEGMENT_BUTTON_COLOR,
    button_primary_background_fill_dark=SEGMENT_BUTTON_COLOR,

    # Clear button
    button_secondary_background_fill=CLEAR_BUTTON_COLOR,
    button_secondary_background_fill_dark=CLEAR_BUTTON_COLOR,

    # Text inside the buttons
    button_primary_text_color=BUTTON_TEXT_COLOR,
    button_primary_text_color_dark=BUTTON_TEXT_COLOR,
    button_secondary_text_color=BUTTON_TEXT_COLOR,
    button_secondary_text_color_dark=BUTTON_TEXT_COLOR,

    # Overlay strength slider
    slider_color=SLIDER_COLOR,
    slider_color_dark=SLIDER_COLOR,
)


# ============================================================
# APPLICATION STYLE
# ============================================================

CSS = f"""
.gradio-container {{
    width: 100% !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px 24px !important;
    box-sizing: border-box !important;
    background-color: {BACKGROUND_COLOR} !important;
}}

#header {{
    text-align: center;
    margin-bottom: 20px;
}}

#header h1 {{
    color: {HEADING_COLOR} !important;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 0;
}}
"""


# ============================================================
# SEGMENTATION FUNCTION
# ============================================================

def segment_image(image, opacity, mode):
    """Run the selected segmentation mode."""

    if image is None:
        raise gr.Error("Please upload an image first.")

    # Send the image, opacity and mode to the segmentation pipeline.
    return run_segmentation(
        image,
        opacity,
        mode
    )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_results():
    """Clear the input image and displayed results."""

    return None, None, None, [], ""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="SceneSense",
    fill_width=True
) as demo:

    # Main application heading.
    gr.Markdown(
        "# SceneSense",
        elem_id="header"
    )

    # --------------------------------------------------------
    # THREE-COLUMN IMAGE LAYOUT
    # --------------------------------------------------------

    with gr.Row(equal_height=False):

        # COLUMN 1: INPUT IMAGE AND CONTROLS

        with gr.Column(
            scale=1,
            min_width=260
        ):

            # Upload an image or capture one using the webcam.
            input_image = gr.Image(
                type="pil",
                label="Upload Photograph",
                height=350,
                sources=["upload", "webcam"]
            )

            # Choose one of the three segmentation modes.
            mode_selector = gr.Radio(
                choices=[
                    SEMANTIC_MODE,
                    INSTANCE_MODE,
                    PANOPTIC_MODE
                ],
                value=SEMANTIC_MODE,
                label="Segmentation Mode"
            )

            # Control the strength of the segmentation overlay.
            opacity_slider = gr.Slider(
                minimum=0.0,
                maximum=0.8,
                value=0.35,
                step=0.05,
                label="Overlay Strength"
            )

            # Place the two buttons next to each other.
            with gr.Row():

                segment_button = gr.Button(
                    "Segment Image",
                    variant="primary"
                )

                clear_button = gr.Button(
                    "Clear",
                    variant="secondary"
                )

        # COLUMN 2: SEGMENTATION MASK

        with gr.Column(
            scale=1,
            min_width=260
        ):

            # Display the coloured mask returned by the pipeline.
            mask_output = gr.Image(
                type="pil",
                label="Segmentation Mask",
                height=350,
                interactive=False
            )

        # COLUMN 3: SEGMENTATION OVERLAY

        with gr.Column(
            scale=1,
            min_width=260
        ):

            # Display the original photo with segmentation colours.
            overlay_output = gr.Image(
                type="pil",
                label="Segmentation Overlay",
                height=350,
                interactive=False
            )

    # --------------------------------------------------------
    # SEGMENTATION RESULTS
    # --------------------------------------------------------

    gr.Markdown("### Segmentation Results")

    # Display image size, detected categories and inference time.
    summary_output = gr.Markdown()

    # Display detected categories, pixel counts and area percentages.
    class_table = gr.Dataframe(
        headers=[
            "Category / Object (Score)",
            "Pixels",
            "Area (%)"
        ],
        datatype=[
            "str",
            "number",
            "number"
        ],
        row_count=0,
        column_count=3,
        interactive=False,
        label="Detected Categories / Objects",
        max_height=350
    )

    # --------------------------------------------------------
    # BUTTON EVENTS
    # --------------------------------------------------------

    # Run segmentation when Segment Image is clicked.
    segment_button.click(
        fn=segment_image,
        inputs=[
            input_image,
            opacity_slider,
            mode_selector
        ],
        outputs=[
            mask_output,
            overlay_output,
            class_table,
            summary_output
        ]
    )

    # Clear all displayed results when Clear is clicked.
    clear_button.click(
        fn=clear_results,
        inputs=[],
        outputs=[
            input_image,
            mask_output,
            overlay_output,
            class_table,
            summary_output
        ]
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    # Process one queued request at a time.
    demo.queue(
        default_concurrency_limit=1
    )

    # Launch SceneSense on the local computer.
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=THEME,
        css=CSS,
        footer_links=[]
    )
