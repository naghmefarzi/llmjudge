from prompts import get_prompt
import torch
from tqdm import tqdm


MAX_LENGTH = 8000

system_message = """You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:

3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query

Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""

def process_batch_quantized(batch, qid_to_query, docid_to_doc,  result_file, model, tokenizer,problematic_passages_path: Optional[str]):

    for eachline in batch.itertuples(index=True):
        try:
            qidx = eachline.qid
            docidx = eachline.docid
            prompt = get_prompt(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=model)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("")
            ]
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators[0],
                pad_token_id=128009,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            
            response = outputs[0][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            
            result_file.write(f"{qidx} 0 {docidx} {response_text}\n")
        
        except Exception as e:
            err = f"Error processing QID {qidx}, DOCID {docidx}: {e}\n"
            # print(err)
            if problematic_passages_path:
                with open(problematic_passages_path,"a") as f:
                    f.write(err)
        
        torch.cuda.empty_cache()  # Clear GPU cache if using GPU  
    
def process_test_qrel_quantized(test_qrel, docid_to_doc, qid_to_query, result_path, model, tokenizer,chunk_size = 100):
    # Open file to write results
    with open(result_path, 'w') as result_file:
        for start_idx in tqdm(range(0, len(test_qrel), chunk_size)):
            # print(start_idx)
            batch = test_qrel.iloc[start_idx:start_idx + chunk_size]
            process_batch_quantized(batch, qid_to_query, docid_to_doc,  result_file, model, tokenizer)
            torch.cuda.empty_cache()  # Clear GPU cache if using GPU  
            del batch
            

