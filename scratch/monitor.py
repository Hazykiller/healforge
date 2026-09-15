import os
import glob
import json

sessions = sorted(glob.glob('workspace/*/session.json'), key=os.path.getmtime, reverse=True)[:15]
for s in sessions:
    try:
        with open(s, encoding='utf-8') as f:
            d = json.load(f)
        pr = d.get('pr', {})
        repo = pr.get('repo', {})
        repo_name = repo.get('full_name', '?') if isinstance(repo, dict) else '?'
        num = pr.get('number', '?')
        title = pr.get('title', '')[:45]
        repairs = len(d.get('repairs', []))
        diag = "YES" if "diagnosis" in d else "NO"
        print(f"PR #{num} ({repo_name}): {title} | diag={diag} | repairs={repairs}")
    except Exception as e:
        pass
