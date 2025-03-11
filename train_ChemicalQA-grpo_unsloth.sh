# 如果你要限制计算卡编号，请在这里设置
export CUDA_VISIBLE_DEVICES=0
export API_KEY="sk-562ba2d5979446b8924fabff4b8600f4"

python train_ChemicalQA-grpo_unsloth.py --config ChemicalQA-grpo_unsloth.yaml

sleep 300 && sudo shutdown -h now
