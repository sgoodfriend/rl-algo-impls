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

python -m pip install --upgrade '.[all]'
python -m pip install -e .