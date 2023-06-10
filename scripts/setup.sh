sudo apt update
sudo apt install -y python-opengl
sudo apt install -y ffmpeg
sudo apt install -y xvfb
sudo apt install -y swig
sudo apt install -y default-jdk

wget https://download.oracle.com/java/19/archive/jdk-19.0.2_linux-x64_bin.deb
sudo apt-get -qqy install ./jdk-19.0.2_linux-x64_bin.deb
sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-19/bin/java 1919
rm jdk-19.0.2_linux-x64_bin.deb

python3 -m pip install --upgrade pip
pip install --upgrade torch torchvision torchaudio

python -m pip install --upgrade '.[all]'