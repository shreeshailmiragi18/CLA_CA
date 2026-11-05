## Quick Start - Run the Complete Project

```bash
# 1. Navigate to project
cd /Users/shreeshailmiragi/Desktop/claude/cla-ca-research-platform

# 2. Install dependencies
pip3 install numpy pandas scipy scikit-learn

# 3. Start the API server
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 4. Start the React dashboard (in another terminal)
cd frontend && npm run dev

# 5. Open dashboard at http://localhost:3000
#    Or API at http://127.0.0.1:8000/docs

# 6. Or run quick evaluation:
python3 -c "
from src.cla_ca.enhancements.e11_hybrid_ensemble import run_e11_hybrid_ensemble
datasets = ['ant-1.5', 'berek', 'jedit-3.2', 'tomcat', 'xalan-2.4', 'EQ', 'JDT']
for ds in datasets:
    r = run_e11_hybrid_ensemble(ds)
    print(f'{ds}: F1={r[\"metrics_ensemble\"][\"f1\"]:.4f}, Prec={r[\"metrics_ensemble\"][\"precision\"]:.3f}, '
          f'Rec={r[\"metrics_ensemble\"][\"recall\"]:.3f}, Pred/Actual={r[\"predicted_defects\"]/max(r[\"actual_defects\"],1):.2f}x')
"
```