import os

def map(qrel_path):
    qid_to_qidx_path = "./private_data/gold_data/qid_to_qidx.txt"
    qid_to_qidx = {}
    with open(qid_to_qidx_path) as qfile:
        qdata = qfile.readlines()
    for line in qdata:
        qidx, qid = line.strip().split("\t")
        qid_to_qidx[qid] = qidx
    
    docid_to_msmarcoid = {}
    docid_to_docidx_path = "./private_data/gold_data/docid_to_docidx.txt"
    
    with open(docid_to_docidx_path) as dfile:
        ddata = dfile.readlines()
    for line in ddata:
        msmarcoid, docid = line.strip().split("\t")
        docid_to_msmarcoid[docid] = msmarcoid
    
    
        
    with open(qrel_path) as f:
        data = f.readlines()
    with open(qrel_path) as f:
        data = f.readlines()
    with open("./private_data/my_qrels/"+qrel_path.replace("./results/","").replace("./private_data/gold_data/",""),"w") as wf:
        for line in data:
            q, _, p, rel = line.strip().split(" ")
            q = qid_to_qidx[q]
            p = docid_to_msmarcoid[p]
            
            print(f"{q} {0} {p} {rel}")
            wf.write(f"{q} {0} {p} {rel}\n")



def main():
    qrel_dir = "./results"
    # Iterate over files in the directory
    for file_name in os.listdir(qrel_dir):
        # Check if the file has a .txt extension
        if file_name.endswith(".txt"):
            qrel_path = os.path.join(qrel_dir, file_name)
            try:
                map(qrel_path)
            except:
                print(f"file {qrel_path} cant be mapped!\n")
                
map("./private_data/gold_data/llm4eval_test_qrel_2024_withRel.txt")