sudo apt update
sudo apt install -y python-opengl
sudo apt install -y ffmpeg
sudo apt install -y xvfb
sudo apt install -y swig

python3 -m pip install --upgrade pip
pip install --upgrade torch torchvision torchaudio

python -m pip install .