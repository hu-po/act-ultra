apt-get update
apt-get -y install neovim x11-server-utils xserver-xorg-video-dummy
python3 -m pip install poetry
poetry install
poetry run pip install torch
cp xorg.conf /etc/X11/xorg.conf
Xorg :1 -config /etc/X11/xorg.conf &
export DISPLAY=:1
