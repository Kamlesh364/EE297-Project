sudo apt update
sudo apt install -y python3-pip python3-venv python3-is-python
python -m venv ee
source ee/bin/activate
pip install -r requirements.txt
sudo apt install nginx
