# export KAGGLE_USERNAME=sgoodfriend
# echo -n "Enter Kaggle Key: "
# read -s KAGGLE_KEY
# mkdir ~/.kaggle
# echo "{
#   \"username\": \"$KAGGLE_USERNAME\",
#   \"key\": \"$KAGGLE_KEY\"
# }" > ~/.kaggle/kaggle.json
# chmod 600 ~/.kaggle/kaggle.json

# Activates poetry shell
bash ./scripts/setup.sh

# Manually install vec-noise within poetry shell
pip install vec-noise

poetry install -E lux
pip install --upgrade "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# kaggle datasets download -d sgoodfriend/lux-replays-flg-npz -p data/lux
# mkdir -p data/lux/lux-replays-flg-npz
# unzip data/lux/lux-replays-flg-npz.zip -d data/lux/lux-replays-flg-npz