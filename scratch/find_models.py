with open('app/ai.py', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
        if 'chat.completions.create' in line or 'extra_body' in line or '"models"' in line:
            print(f'{idx}: {line.strip()}')
