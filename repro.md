\# repro.md



\## Environment



\- Windows 10

\- Python 3.x (venv)

\- PyTorch + Torchvision

\- Device: CPU



\## Commands (exactly what I ran)



cd OneDrive\\Desktop\\JTC\\Week\_16\\W16D2-After-Class

venv\\Scripts\\activate

python layers.py

python train.py



\## Seed



\- `torch.manual\_seed(42)`

\- `torch.cuda.manual\_seed\_all(42)` (conditional, but CPU-only in my run)



\## Dataset



\- FashionMNIST (downloaded automatically)

\- train/val split: 50,000 / 10,000

\- batch\_size = 128



\## Hyperparameters



\- epochs = 3

\- optimizer = AdamW(lr=1e-3)

\- scheduler = StepLR(step\_size=1, gamma=0.8)

\- loss = CrossEntropyLoss

\- device = CPU



\## Notes



Training logs were reproducible across runs with the above seed + setup.



