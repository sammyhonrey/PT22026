"""Custom CSS for the ContextAI Gradio interface."""

CUSTOM_CSS = """
:root {
    --bg: #EAF7FC;
    --panel: #FFFFFF;
    --panel-soft: #EAF7FC;
    --border: #C6E3F1;
    --text: #173B50;
    --muted: #64869A;
    --blue: #87CEEB;
    --purple: #800020;
}

body {
    background: radial-gradient(ellipse at top, #E2F5FC 0%, #f7fbfe 62%) !important;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1420px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 28px clamp(14px, 3vw, 42px) 30px !important;
    background: transparent !important;
}

.gradio-container, .gradio-container .main, .gradio-container .wrap {
    background: transparent !important;
}

#hero {
    text-align: center;
    padding: 10px 0 28px;
}

#hero h1 {
    font-size: clamp(34px, 4vw, 48px) !important;
    font-weight: 850 !important;
    letter-spacing: -1.4px;
    margin: 0 0 8px !important;
    background: linear-gradient(90deg, #87CEEB, #55A9D1, #800020);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#hero .subtitle {
    color: #173B50;
    font-size: 17px;
    font-weight: 650;
    margin-bottom: 8px;
}

#hero .description {
    color: var(--muted);
    font-size: 15px;
}

.panel {
    background: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18) !important;
}

.panel h2, .section-heading {
    color: #173B50 !important;
    font-size: 23px !important;
    font-weight: 750 !important;
    margin: 0 0 8px !important;
}

.panel-description {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 18px;
}

label span, .label {
    color: #31566b !important;
    font-weight: 650 !important;
}

textarea, input, .input-container {
    background: #ffffff !important;
    border: 1px solid #C6E3F1 !important;
    color: #173B50 !important;
    border-radius: 10px !important;
}

textarea:focus, input:focus {
    border-color: #87CEEB !important;
    box-shadow: 0 0 0 2px rgba(138, 186, 211, 0.24) !important;
}

textarea::placeholder, input::placeholder {
    color: #8AAFC3 !important;
}

#context-box textarea {
    min-height: 245px !important;
    line-height: 1.65 !important;
}

#answer-box textarea {
    min-height: 245px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    line-height: 1.7 !important;
    color: #800020 !important;
}

button.primary {
    background: linear-gradient(100deg, #87CEEB, #55A9D1) !important;
    border: 0 !important;
    color: white !important;
    font-weight: 750 !important;
    border-radius: 10px !important;
    min-height: 48px !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}

button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 7px 20px rgba(92, 163, 197, .30) !important;
}

button.secondary {
    background: #800020 !important;
    border: 1px solid #6B1730 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    min-height: 48px !important;
}

#button-row {
    align-items: stretch !important;
    flex-wrap: nowrap !important;
    margin-top: 12px !important;
}

#upload-area {
    background: rgba(234, 245, 251, .85) !important;
    border: 1px dashed #87CEEB !important;
    border-radius: 12px !important;
}

#confidence-panel {
    background: #EAF7FC !important;
    border: 1px solid #C6E3F1 !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
}

.confidence-wrap { padding: 2px 0 0; }
.confidence-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.confidence-caption { color: #64869A; font-size: 13px; font-weight: 600; }
.confidence-value { font-size: 15px; font-weight: 800; white-space: nowrap; }
.confidence-track {
    height: 10px;
    background: #ffffff;
    border: 1px solid #C6E3F1;
    border-radius: 999px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 999px;
    transition: width .25s ease;
}
.confidence-note {
    color: #6e8a9d;
    font-size: 12px;
    margin: 10px 0 0;
}

#examples-section {
    margin-top: 28px !important;
}
#history-panel {
    margin-top: 28px !important;
}
#history-table {
    border: 1px solid #C6E3F1 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
#history-table table {
    background: #ffffff !important;
}
#history-table th {
    background: #DDF2FA !important;
    color: #173B50 !important;
}
#history-table td {
    color: #31566b !important;
}

#about-box {
    border: 1px solid #2e405a !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, .88) !important;
    margin-top: 22px !important;
}

#footer {
    text-align: center;
    color: #64869A;
    font-size: 13px;
    padding: 24px 0 6px;
}
#footer .brand {
    color: #800020;
    font-weight: 750;
}

@media (max-width: 850px) {
    .gradio-container {
        padding: 18px 12px 24px !important;
    }
    .panel {
        padding: 16px !important;
        border-radius: 16px !important;
    }
    #context-box textarea, #answer-box textarea {
        min-height: 190px !important;
    }
    #button-row {
        flex-wrap: wrap !important;
    }
}
"""
