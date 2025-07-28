import json

def load_cuad_qa(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for doc in data['data']:
        contract_id = doc.get('title', 'unknown')
        for paragraph in doc.get('paragraphs', []):
            text = paragraph.get('context', '')
            for qa in paragraph.get('qas', []):
                label = qa.get('question', '')
                for ans in qa.get('answers', []):
                    start = ans.get('answer_start', None)
                    clause = ans.get('text', '').strip()
                    if clause and start is not None:
                        entries.append({
                            'contract_id': contract_id,
                            'clause': clause,
                            'label': label,
                            'start': start
                        })
    return entries
