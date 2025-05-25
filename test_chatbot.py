import streamlit as st
import json
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration & Data ---

# Prompt templates from run_longbench.py
MODEL2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
}

# Expected max new tokens for generation, based on dataset2maxlen from run_longbench.py
DATASET2MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    # Add other datasets and their expected max output lengths if needed
}

# Placeholder for model path - user should change this to a valid Hugging Face model
DEFAULT_MODEL_PATH = "gpt2" # A small model for quick testing, e.g., gpt2, gpt2-medium
# For actual benchmarking, you'd use models like "meta-llama/Llama-2-7b-chat-hf", etc.

# --- Helper Functions ---

@st.cache_resource
def load_model_and_tokenizer(model_path):
    """Loads the model and tokenizer, and caches them."""
    st.info(f"Attempting to load model: {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id # Ensure pad_token_id is also set

        # Set device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set model to evaluation mode
        st.success(f"Model '{model_path}' loaded successfully on {device}!")
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model/tokenizer for '{model_path}': {e}")
        return None, None, None

def load_dataset(dataset_name):
    """
    Loads a dataset from a JSONL file.
    Assumes files are in a './data/' directory (e.g., './data/narrativeqa.jsonl').
    """
    file_path = f"./data/{dataset_name}.jsonl"
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        if not data:
            st.warning(f"Dataset file {file_path} is empty. Using dummy data for {dataset_name}.")
            return get_dummy_data(dataset_name)
        return data
    except FileNotFoundError:
        st.warning(f"Dataset file {file_path} not found. Using dummy data for {dataset_name}.")
        return get_dummy_data(dataset_name)
    except Exception as e:
        st.error(f"Error loading dataset {dataset_name}: {e}")
        return get_dummy_data(dataset_name) # Fallback to dummy data on other errors

def get_dummy_data(dataset_name):
    """Provides dummy data if actual dataset files are missing or empty."""
    if dataset_name == "narrativeqa":
        return [
            {"_id": "dummy_nq_1", "context": "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversation?'", "input": "Why was Alice tired?", "answers": ["sitting by her sister on the bank, and of having nothing to do"], "length": 50},
            {"_id": "dummy_nq_2", "context": "The March Hare and the Hatter were having tea at it: a Dormouse was sitting between them, fast asleep, and the other two were using it as a cushion, resting their elbows on it, and talking over its head.", "input": "Who was fast asleep?", "answers": ["The Dormouse"], "length": 40},
        ]
    elif dataset_name == "qasper":
        return [
            {"_id": "dummy_qsp_1", "context": "The study investigates the impact of climate change on bird migration patterns. Results indicate a significant shift in migration timing for several species.", "input": "What do the results indicate?", "answers": ["a significant shift in migration timing for several species"], "length": 25},
            {"_id": "dummy_qsp_2", "context": "CRISPR-Cas9 is a gene editing technology that allows scientists to alter DNA sequences. It has potential applications in treating genetic disorders.", "input": "What is a potential application of CRISPR-Cas9?", "answers": ["treating genetic disorders"], "length": 30},
        ]
    return []


def truncate_text(text, max_display_length=200):
    """Truncates text for display purposes and adds ellipsis."""
    return (text[:max_display_length] + "...") if len(text) > max_display_length else text

# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("📝 LLM Benchmark Chatbot")
st.markdown("Test Hugging Face models on selected datasets and view performance metrics.")

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")
selected_model_path = st.sidebar.text_input("Hugging Face Model Path", value=DEFAULT_MODEL_PATH, help="E.g., gpt2, meta-llama/Llama-2-7b-chat-hf")

# Attempt to load model and tokenizer
model, tokenizer, device = None, None, None
if selected_model_path:
    model, tokenizer, device = load_model_and_tokenizer(selected_model_path)

available_datasets = list(MODEL2PROMPT.keys())
selected_dataset_name = st.sidebar.selectbox("Select Dataset", available_datasets, index=0)

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Prompt Template")
if selected_dataset_name:
    # Display prompt template with placeholders for clarity
    prompt_template_display = MODEL2PROMPT[selected_dataset_name].replace("{context}", "{CONTEXT_PLACEHOLDER}").replace("{input}", "{INPUT_PLACEHOLDER}")
    st.sidebar.text_area("Template:", prompt_template_display, height=280, disabled=True, help="This is the template used to construct the input for the model.")


# --- Main Area for Testing ---
st.header("🚀 Run Benchmark Test")

# Initialize session state variables
if 'current_sample' not in st.session_state:
    st.session_state.current_sample = None
if 'results' not in st.session_state:
    st.session_state.results = None

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample & Control")
    if st.button("♻️ Load Random Sample & Test", type="primary", disabled=not (model and tokenizer and device)):
        st.session_state.results = None # Clear previous results before new test
        dataset = load_dataset(selected_dataset_name)

        if not dataset:
            st.error(f"No data could be loaded for the '{selected_dataset_name}' dataset. Please ensure the data file exists at `./data/{selected_dataset_name}.jsonl` or check dummy data.")
        else:
            st.session_state.current_sample = random.choice(dataset)
            sample = st.session_state.current_sample

            prompt_template = MODEL2PROMPT[selected_dataset_name]
            full_prompt = prompt_template.format(context=sample.get("context", ""), input=sample.get("input", ""))

            # Prepare display for input
            input_display_parts = [
                f"**Dataset:** `{selected_dataset_name}`",
                f"**Sample ID:** `{sample.get('_id', 'N/A')}`",
                f"**Context (truncated for display):**\n```\n{truncate_text(sample.get('context', ''), 300)}\n```",
                f"**Question/Input:** `{sample.get('input', '')}`"
            ]
            st.session_state.results = {"input_display": "\n\n".join(input_display_parts)}

            with st.spinner(f"Processing with '{selected_model_path}' on {device}... This may take a moment."):
                try:
                    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length - DATASET2MAXLEN.get(selected_dataset_name, 100) - 10, padding=True) # Ensure space for generation + buffer
                    inputs = inputs.to(device)
                    input_ids = inputs.input_ids
                    attention_mask = inputs.attention_mask # Get attention mask
                    actual_input_token_count = input_ids.shape[1]

                    max_new_tokens = DATASET2MAXLEN.get(selected_dataset_name, 100) # Get max new tokens based on dataset

                    time_start_generate = time.perf_counter()
                    # Generate response using the model
                    generated_outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask, # Pass attention_mask
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        # For more deterministic output, consider:
                        # num_beams=1,
                        # do_sample=False,
                    )
                    time_end_generate = time.perf_counter()

                    # Extract only the newly generated tokens
                    generated_token_ids = generated_outputs[0][actual_input_token_count:]
                    model_answer_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    num_generated_tokens = len(generated_token_ids)

                    # Calculate performance metrics
                    latency_seconds = time_end_generate - time_start_generate
                    tokens_per_second = (num_generated_tokens / latency_seconds) if latency_seconds > 0 and num_generated_tokens > 0 else 0.0

                    # Store all results for display
                    st.session_state.results.update({
                        "model_answer": model_answer_text,
                        "ground_truth": "\n".join(sample["answers"]) if isinstance(sample.get("answers"), list) else sample.get("answers", "N/A"),
                        "context_length_tokens": actual_input_token_count, # Actual tokens in the input prompt
                        "generated_tokens": num_generated_tokens,
                        "latency_sec": f"{latency_seconds:.3f}",
                        "tokens_per_sec": f"{tokens_per_second:.2f}",
                    })
                    st.success("Processing complete!")

                except Exception as e:
                    st.error(f"Error during model generation: {e}")
                    st.session_state.results = {"error": str(e)} # Store error for display
    else:
        if not (model and tokenizer and device):
            st.warning("Please enter a valid model path in the sidebar to enable testing.")


# --- Display Area for Sample and Results ---
# Display Input Sample (always in col1)
if st.session_state.results and "input_display" in st.session_state.results:
    st.markdown(st.session_state.results["input_display"])
elif st.session_state.results and "error" in st.session_state.results and not "input_display" in st.session_state.results: # If error happened before input display was set
    st.error(f"Could not prepare sample due to error: {st.session_state.results['error']}")
elif not st.session_state.results:
     st.info("Click 'Load Random Sample & Test' to begin.")


with col2:
    st.subheader("📊 Results")
    if st.session_state.results:
        if "error" in st.session_state.results:
            st.error(f"Processing Failed: {st.session_state.results['error']}")
        elif "model_answer" in st.session_state.results: # Successfully processed
            st.markdown(f"**🤖 Model's Answer:**")
            st.info(st.session_state.results["model_answer"])

            st.markdown(f"**🎯 Ground Truth:**")
            st.success(st.session_state.results["ground_truth"])

            st.markdown("---")
            st.markdown(f"**🔢 Input Prompt Tokens:** `{st.session_state.results['context_length_tokens']}`")
            st.markdown(f"**💡 Generated Tokens:** `{st.session_state.results['generated_tokens']}`")
            st.markdown(f"**⏱️ Latency:** `{st.session_state.results['latency_sec']} seconds`")
            st.markdown(f"**⚡ Tokens/Second (Decode):** `{st.session_state.results['tokens_per_sec']}`")
        # No specific message if only input_display is set, as main call to action is button.
    else:
        st.info("Results will appear here after testing.")


# --- Instructions / Notes ---
st.markdown("---")
st.markdown("""
**📖 How to Use:**
1.  **Enter Model Path:** In the sidebar, input the Hugging Face model identifier (e.g., `gpt2`, `meta-llama/Llama-2-7b-chat-hf`). The application will attempt to load it.
2.  **Select Dataset:** Choose between `narrativeqa` or `qasper` from the dropdown. The prompt template used for that dataset will be shown.
3.  **Load & Test:** Click the "♻️ Load Random Sample & Test" button.
    * A random sample (context & question) from the chosen dataset will be selected.
    * The model will generate an answer.
    * The input, model's answer, ground truth, and performance metrics (input token length, generated tokens, latency, tokens/second) will be displayed.

**📝 Notes:**
* **Prerequisites:** Ensure you have `torch`, `transformers`, and `streamlit` Python packages installed.
* **GPU Usage:** The script automatically uses a GPU if `torch.cuda.is_available()` is true. Otherwise, it falls back to CPU.
* **Data Files:** This app expects dataset files (e.g., `narrativeqa.jsonl`, `qasper.jsonl`) in a `./data/` subdirectory relative to where you run the script. If not found, it will use built-in dummy data.
    * Each line in a `.jsonl` file should be a JSON object. Required keys: `"context"` (string), `"input"` (string), `"answers"` (list of strings or a single string). Optional but good: `"_id"` (string).
* **Tokenization & Max Length:** Input is truncated to fit within the model's maximum context length, reserving space for generated tokens based on `DATASET2MAXLEN`.
* **Performance Metrics:**
    * `Input Prompt Tokens`: Number of tokens in the formatted input (context + question) fed to the model.
    * `Generated Tokens`: Number of new tokens produced by the model.
    * `Latency`: Total time (in seconds) for the `model.generate()` call.
    * `Tokens/Second`: Calculated as `Generated Tokens / Latency`. This primarily reflects decoding speed for the generated part.
""", unsafe_allow_html=True)