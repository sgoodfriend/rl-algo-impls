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
poetry run pip install vec-noise

poetry install -E lux

# Detect CUDA version
cuda_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')

# Check if CUDA version is 11 or 12 and install the appropriate jax version
if [[ $cuda_version == 11.* ]]; then
    echo "CUDA version 11 detected. Installing jax for CUDA 11."
    poetry run pip install --upgrade "jax[cuda11_pip]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [[ $cuda_version == 12.* ]]; then
    echo "CUDA version 12 detected. Installing jax for CUDA 12."
    poetry run pip install --upgrade "jax[cuda12_pip]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
    echo "Unsupported CUDA version: $cuda_version"
fi


# kaggle datasets download -d sgoodfriend/lux-replays-flg-npz -p data/lux
# mkdir -p data/lux/lux-replays-flg-npz
# unzip data/lux/lux-replays-flg-npz.zip -d data/lux/lux-replays-flg-npz