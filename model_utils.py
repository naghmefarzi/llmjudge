from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from huggingface_hub import login
from transformers import (AutoTokenizer,AutoModelForCausalLM,
                          TextStreamer,pipeline,BitsAndBytesConfig)
import transformers
import torch

def get_model_baseline(name_or_path_to_model : str):
    

    pipeline = transformers.pipeline(
        "text-generation",
        model=name_or_path_to_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def get_model_quantized(name_or_path_to_model: str) -> Tuple:


    tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path_to_model,
        #torch_dtype=torch.bfloat16,
        quantization_config = bnb_config,
        device_map="auto",
    )
    return model,tokenizer
