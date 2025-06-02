"""
此檔案已轉換為使用 PyMARL 框架執行 QMix 訓練。

請依以下步驟操作：

1. 安裝 PyMARL2
----------------
git clone https://github.com/hijkzzz/pymarl2.git
cd pymarl2
conda create -n pymarl2 python=3.8 -y
conda activate pymarl2
pip install -r requirements.txt

2. 自訂 reward function
------------------------
請至 `src/envs/starcraft2/starcraft2.py` 中修改 `_reward_battle` 或 `_reward_health` 函數，
以實現你自己的 reward 設計邏輯。例如：

def _reward_battle(self):
    # 自訂獎勵邏輯
    reward = ...
    return reward

3. 開始訓練 QMix
----------------
cd pymarl2
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m

4. 檢視結果
----------------
輸出結果與模型會儲存在 `results/` 資料夾中，包含 TensorBoard log 與 checkpoints。
"""
