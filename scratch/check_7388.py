import json
import glob

for path in glob.glob('workspace/*/session.json'):
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        pr = data.get('pr', {})
        if str(pr.get('number')) == '7388':
            print('FOUND:', path)
            print('PR Title:', pr.get('title'))
            repairs = data.get('repairs', [])
            print('Repairs count:', len(repairs))
            for i, r in enumerate(repairs):
                print(f"=== Repair {i+1} (attempt {r.get('attempt')}) ===")
                print("Explanation:", r.get("explanation"))
                print("Patch:\n", r.get("patch"))
            verifs = data.get('verifications', [])
            print('Verifications count:', len(verifs))
            for j, v in enumerate(verifs):
                print(f"=== Verification {j+1} (attempt {v.get('attempt')}) ===")
                print("Passed:", v.get("passed"))
                print("Reason:", v.get("reason"))
                print("Output:\n", v.get("output"))
    except Exception as e:
        pass
