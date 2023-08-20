import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_base_model(base_model_name):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'
    )
    base_model.save_pretrained("./components/model-relocation-server/models/" + base_model_name)
    torch.save(base_model.state_dict(),
               "./components/model-relocation-server/models/" + base_model_name + "/" + base_model_name + ".pt")


def download_base_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained("./components/tokenizer-relocation-server/tokenizers/" + tokenizer_name)


def download_reference_model(reference_model_name):
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'
    )
    reference_model.save_pretrained("./components/model-reference-relocation-server/models/" + reference_model_name)
    torch.save(reference_model.state_dict(),
               "./components/model-reference-relocation-server/models/" + reference_model_name + "/" + reference_model_name + ".pt")

def download_reference_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained("./components/model-reference-relocation-server/tokenizers/" + tokenizer_name)


if __name__ == '__main__':

    if os.getcwd().endswith('/scripts'):
        print("Please execute the script from the base project directory")
        exit(1)

    download_base_model("gpt2")
    download_base_tokenizer("gpt2")
    download_reference_model("gpt2-large")
    download_reference_tokenizer("gpt2-large")
