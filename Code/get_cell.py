import json
import glob
notebooks = glob.glob('*.ipynb')
for nb in notebooks:
    try:
        n = json.load(open(nb))
        cells = [c['source'] for c in n['cells'] if 'cell_type' in c and c['cell_type'] == 'code']
        for c in cells:
            s = ''.join(c)
            if 'pf_ode' in s or 'euler_maruyama' in s:
                print(f"--- {nb} ---")
                print(s)
                print("-" * 40)
    except Exception as e: pass
