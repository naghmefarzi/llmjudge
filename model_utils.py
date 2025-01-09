from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from huggingface_hub import login
from transformers import (AutoTokenizer,AutoModelForCausalLM,
                          TextStreamer,pipeline,BitsAndBytesConfig,AutoModelForSeq2SeqLM)

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import transformers
import torch
from together_model import TogetherPipeline

# def get_model_baseline(name_or_path_to_model : str):
    

#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=name_or_path_to_model,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="auto",
#     )
#     return pipeline



def get_model_baseline(name_or_path_to_model: str, use_together: bool = False):
    
    if use_together:
        # Initialize TogetherPipeline for API-based interaction
        return TogetherPipeline(model_name=name_or_path_to_model)
    # Check if the model is Flan-T5 based on its name
    elif "flan-t5" in name_or_path_to_model.lower():
        # Load Flan-T5 as a sequence-to-sequence model
        tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            name_or_path_to_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Use the text2text-generation pipeline
        text_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer
        )
    else:
        # Assume it's a causal language model (like LLaMA)
        tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path_to_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Use the text-generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    
    return text_pipeline


# def get_model_quantized(name_or_path_to_model: str) -> Tuple:


#     tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
#     bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=torch.float16,
#             )
#     model = AutoModelForCausalLM.from_pretrained(
#         name_or_path_to_model,
#         #torch_dtype=torch.bfloat16,
#         quantization_config = bnb_config,
#         device_map="auto",
#     )
#     return model,tokenizer
