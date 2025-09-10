# Ubuntu Installation & Configuration


## Installation

### Connect to VM



### Update VM

```Code

sudo apt list --upgradable
sudo apt update
sudo apt upgrade

```

### Python Installation 

```Code
python3 -V
Python 3.12.3
```

Next, Install Pip using beneath command.

```Code
sudo apt install -y python3-pip
sudo apt install python3.12-venv
```

### Python Configuration

```Code

python3 -m venv .venv && source .venv/bin/activate

```

```Code

pip3 install --upgrade pip
pip3 install oracledb pdfplumber pymupdf pillow pytesseract blingfire
pip3 install sentence-transformers  # if EMBED_PROVIDER=local
pip3 install cohere                 # if EMBED_PROVIDER=oci (Cohere-compatible)

```






