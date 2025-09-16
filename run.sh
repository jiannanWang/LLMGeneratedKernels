uv venv
source .venv/bin/activate
git clone git@github.com:meta-pytorch/BackendBench.git
uv pip install BackendBench/
uv pip install matplotlib

cd nanoGPT
cd data/shakespeare_char && uv run python prepare.py && cd ../..

mkdir -p logs

uv run python train.py config/train_shakespeare_char.py --device=cuda --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --use_backendbench=True --kernel_folder="../generated_kernels_opinfo" > logs/backendbench_log.txt 2>&1
uv run python train.py config/train_shakespeare_char.py --device=cuda --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 > logs/log.txt 2>&1

uv run python draw_curves.py
cd ..