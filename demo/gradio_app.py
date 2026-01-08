import gradio as gr
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.analyzer import NLPAnalyzer

# Load models once when server starts
print("Loading models...")
analyzer = NLPAnalyzer()
print("Ready! Starting web server...")


# -------- NLP FUNCTIONS --------
def sentiment_tab(text):
    if not text.strip():
        return "Enter some text first"
    result = analyzer.analyze_sentiment(text)
    return f"**{result['label']}** (confidence: {result['confidence']})"


def ner_tab(text):
    if not text.strip():
        return "Enter some text first"
    result = analyzer.extract_entities(text)
    if not result["entities"]:
        return "No entities found"
    
    output = []
    for ent in result["entities"]:
        output.append(f"- **{ent['text']}** [{ent['type']}]")
    return "\n".join(output)


def zeroshot_tab(text, categories):
    if not text.strip():
        return "Enter some text first"
    if not categories.strip():
        return "Enter at least one category"

    cat_list = [c.strip() for c in categories.split(",")]
    result = analyzer.classify_zero_shot(text, cat_list)

    output = [f"**Winner: {result['top_category']}**\n"]
    for item in result["all_scores"]:
        bar = "█" * int(item["score"] * 20)
        output.append(f"{item['category']}: {bar} {item['score']}")
    return "\n".join(output)


def summary_tab(text):
    if not text.strip():
        return "Enter some text first"

    if len(text.split()) < 50:
        return "Enter longer text (50+ words) for summarization"

    result = analyzer.summarize(text)
    return (
        f"**Summary:**\n\n{result['summary']}\n\n"
        f"---\n*{result['original_words']} words → {result['summary_words']} words*"
    )


# -------- BUTTON STATE HELPERS --------
def set_analyzing():
    return gr.update(value="Analyzing...", interactive=False)

def reset_analyze():
    return gr.update(value="Analyze", interactive=True)


def set_extracting():
    return gr.update(value="Extracting...", interactive=False)

def reset_extract():
    return gr.update(value="Extract", interactive=True)


def set_classifying():
    return gr.update(value="Classifying...", interactive=False)

def reset_classify():
    return gr.update(value="Classify", interactive=True)


def set_summarizing():
    return gr.update(value="Summarizing...", interactive=False)

def reset_summarize():
    return gr.update(value="Summarize", interactive=True)


        #  UI 
with gr.Blocks(title="NLP Intelligence Hub") as app:
    gr.Markdown("# NLP Intelligence Hub")
    gr.Markdown("Multi-task text analysis powered by Hugging Face Transformers")

    with gr.Tabs():

        #  SENTIMENT 
        with gr.Tab("Sentiment"):
            gr.Markdown("### Is the text positive or negative?")
            s_in = gr.Textbox(label="Text", lines=3)
            s_out = gr.Markdown()
            # Event Chaining
            s_btn = gr.Button("Analyze")

            (
                s_btn
                .click(set_analyzing, outputs=s_btn)
                .then(sentiment_tab, inputs=s_in, outputs=s_out)
                .then(reset_analyze, outputs=s_btn)
            )

        # -------- NER --------
        with gr.Tab("Named Entities"):
            gr.Markdown("### Find people, organizations, locations")
            n_in = gr.Textbox(label="Text", lines=3)
            n_out = gr.Markdown()
            n_btn = gr.Button("Extract")

            (
                n_btn
                .click(set_extracting, outputs=n_btn)
                .then(ner_tab, inputs=n_in, outputs=n_out)
                .then(reset_extract, outputs=n_btn)
            )

        # -------- ZERO-SHOT --------
        with gr.Tab("Zero-Shot Classification"):
            gr.Markdown("### Classify into ANY categories (no training needed!)")
            z_text = gr.Textbox(label="Text", lines=3)
            z_cats = gr.Textbox(
                label="Categories (comma-separated)",
                value="finance, sports, politics, technology",
            )
            z_out = gr.Markdown()
            z_btn = gr.Button("Classify")

            (
                z_btn
                .click(set_classifying, outputs=z_btn)
                .then(zeroshot_tab, inputs=[z_text, z_cats], outputs=z_out)
                .then(reset_classify, outputs=z_btn)
            )

        # -------- SUMMARY --------
        with gr.Tab("Summarization"):
            gr.Markdown("### Condense long text into a summary")

            sum_in = gr.Textbox(
                label="Long text (50+ words)",
                placeholder="Paste a long article here...",
                lines=8,
            )

            sum_out = gr.Markdown()
            sum_btn = gr.Button("Summarize")

            (
                sum_btn
                .click(set_summarizing, outputs=sum_btn)
                .then(summary_tab, inputs=sum_in, outputs=sum_out)
                .then(reset_summarize, outputs=sum_btn)
            )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

