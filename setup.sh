sudo apt update
sudo apt install -y python3-pip python3-venv python-is-python3
python -m venv ee
source ee/bin/activate
pip install -r requirements.txt
sudo apt install python3-opencv
sudo apt install nginx
