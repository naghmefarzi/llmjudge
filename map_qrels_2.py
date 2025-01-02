def map(qrel_path):
    qid_to_qidx_path = "./private_data/gold_data/qid_to_qidx.txt"
    qid_to_qidx = {}
    with open(qid_to_qidx_path) as qfile:
        qdata = qfile.readlines()
    for line in qdata:
        qidx, qid = line.split("\t")
        qid_to_qidx[qid.strip()] = qidx.strip()
    
    docid_to_msmarcoid = {}
    docid_to_docidx_path = "./private_data/gold_data/docid_to_docidx.txt"
    
    with open(docid_to_docidx_path) as dfile:
        ddata = dfile.readlines()
    for line in ddata:
        msmarcoid, docid = line.split("\t")
        docid_to_msmarcoid[docid.strip()] = msmarcoid.strip()
    
    
        
    with open(qrel_path) as f:
        data = f.readlines()
    with open(qrel_path) as f:
        data = f.readlines()
    with open("./private_data/my_qrels"+qrel_path.replace("./results/",""),"w") as wf:
        for line in data:
            q, _, p, rel = line.split(" ")
            q = qid_to_qidx[q]
            p = docid_to_msmarcoid[p]
            wf.write(f"{q} {0} {p} {rel}")

qrel_path = "./results/dev_baseline_qrel_prompt order: 0123.txt"
map(qrel_path)