while test $# != 0
do
    case "$1" in
        --microrts) microrts=t ;;
        --lux) lux=t ;;
    esac
    shift
done

sudo apt update
sudo apt install -y python-opengl
sudo apt install -y ffmpeg
sudo apt install -y xvfb
sudo apt install -y swig
sudo apt install -y default-jdk

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install -y git-lfs
git lfs pull

python3 -m pip install --upgrade pip
pip install --upgrade torch torchvision torchaudio

python3 -m pip install --upgrade pipx
python3 -m pipx ensurepath --force
pipx install poetry

poetry run pip install --upgrade pip

poetry_extras=""
if [ "$microrts" = "t" ]; then
    poetry_extras+=" -E microrts"
fi
if [ "$lux" = "t" ]; then
    poetry run pip install vec-noise
    poetry_extras+=" -E lux"
fi
poetry install $poetry_extras

if [ "$lux" = "t" ]; then
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
fi

poetry run wandb login