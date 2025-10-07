# Small LLM Fine-Tuning using DistilGPT-2

This project demonstrates how to fine-tune a lightweight Large Language Model (LLM) — specifically DistilGPT-2, a distilled and faster version of GPT-2 — on a custom dataset using Hugging Face Transformers.
It’s ideal for learning, experimentation, and small-scale language modeling tasks.

🧠 Overview

This notebook walks through:

Setting up the environment

Loading and tokenizing your custom dataset

Fine-tuning a pre-trained small LLM (DistilGPT-2)

Saving and testing the fine-tuned model

You’ll end up with a custom-trained text generation model that learns patterns from your dataset and can generate similar outputs.

📦 Requirements

Install all dependencies before running the notebook:

!pip install transformers datasets torch accelerate


✅ Recommended: Run this on Google Colab with a GPU runtime enabled (Runtime → Change runtime type → GPU).

⚙️ Project Structure
Cell	Description
Cell 1	Install required libraries
Cell 2	Import all dependencies
Cell 3	Load dataset (custom text file or Hugging Face dataset)
Cell 4	Preprocess and tokenize data
Cell 5	Prepare data for model input
Cell 6	Load DistilGPT-2 model and tokenizer
Cell 7	Define training parameters using TrainingArguments
Cell 8	Initialize the Trainer
Cell 9	Train and save the model (fine-tuning step)
Cell 10	Load the fine-tuned model and test it with sample prompts
🧰 Model Used

DistilGPT-2

A distilled (compressed) version of OpenAI’s GPT-2

Has ~82M parameters (smaller and faster than GPT-2’s 124M)

Great for low-resource environments and quick fine-tuning

🧪 Testing the Model

Once fine-tuning is complete, you can test your model like this:

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./finetuned-llm-distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

🕒 Training Time

Training time depends on:

Dataset size

Number of epochs

GPU availability

For small datasets (~few KB), fine-tuning usually takes 3–10 minutes on a Colab GPU.

💾 Saving & Loading

The model and tokenizer are saved locally at:

./finetuned-llm-distilgpt2/


You can reload them anytime for inference or further fine-tuning.

🔍 Example Use Cases

Text generation or story completion

Chatbot response modeling

Domain-specific writing (e.g., law, medicine, tech blogs)

Prototype-level LLM customization

📚 References

Hugging Face Transformers Documentation

DistilGPT-2 Model Card

Google Colab Guide
