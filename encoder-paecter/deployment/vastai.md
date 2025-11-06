Copy requirements and install PyTorch with CUDA 12.1 support.


```
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

## Run in the background

```
nohup python -u main.py > worker.log 2>&1 &
```