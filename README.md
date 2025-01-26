# pooling_paper
GitHub do Paper: Pooling is All You Need

# Tutorial

git clone https://github.com/yan-soares/pooling_paper.git
cd pooling_paper
conda create -n pooling_paper_env python==3.11
conda activate pooling_paper_env

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install scikit-learn
pip install transformers
pip install sympy==1.13.1
pip install pandas
pip install sentencepiece

#OR

pip install -r requirements.txt

# Exemplo de execução
python main_experiments.py --task_type classification --models deberta-base --epochs 1 --batch 512 --kfold 10 --optim adam --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL


