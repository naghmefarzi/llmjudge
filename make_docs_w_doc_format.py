import json


path = "./data/dl2019/trecDL2019-qrels-runs-with-text.jsonl"
path = "./data/dl2020/trecDL2020-qrels-runs-with-text.jsonl"

docids=[]
all_data = []
with open(path,"r") as f:
    for line in f:
        data = json.loads(line)
        res = data[1]
        for entry in res:
            docid = entry["paragraph_id"]
            doc = entry['text']
            if docid not in docids:
                jsonline = {
                    "docid":docid,
                    "doc":doc
                }
                docids.append(docid)
                all_data.append(jsonline)
        



with open('./data/dl2020/dl2020_document.jsonl', 'w') as file:
    for entry in all_data:
        # Convert dictionary to JSON and write it as a line
        file.write(json.dumps(entry) + '\n')