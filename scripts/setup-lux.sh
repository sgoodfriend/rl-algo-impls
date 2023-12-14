export KAGGLE_USERNAME=sgoodfriend
echo -n "Enter Kaggle Key: "
read -s KAGGLE_KEY
mkdir ~/.kaggle
echo "{
  \"username\": \"$KAGGLE_USERNAME\",
  \"key\": \"$KAGGLE_KEY\"
}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

bash ./scripts/setup.sh
python -m pip install vec-noise
poetry install -E lux

# kaggle datasets download -d sgoodfriend/lux-replays-flg-npz -p data/lux
# mkdir -p data/lux/lux-replays-flg-npz
# unzip data/lux/lux-replays-flg-npz.zip -d data/lux/lux-replays-flg-npz
wandb login