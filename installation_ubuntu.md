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


### Tesseracr Installation

```Code
sudo add-apt-repository ppa:alex-p/tesseract-ocr5

sudo apt update

sudo apt install tesseract-ocr
```

### Install PODMAN

```Code
sudo apt update
sudo apt list --upgradable
sudo apt install podman -y
sudo podman --version
sudo systemctl enable podman.socket
sudo systemctl start podman.socket
sudo systemctl stop podman.socket
sudo systemctl restart podman.socket
sudo systemctl status podman.socket
podman images
```

```Code

podman pull container-registry.oracle.com/database/free:latest
podman pull docker.io/ollama/ollama

```

### Configure open ports

```Code
sudo iptables -L
sudo vi /etc/iptables/rules.v4
```

Add these lines in this file

-A INPUT -p tcp -m state --state NEW -m tcp --dport 8501 -j ACCEPT
-A INPUT -p tcp -m state --state NEW -m tcp --dport 11234 -j ACCEPT

```Code
sudo iptables-restore < /etc/iptables/rules.v4
```

### Configure Oracle database locally


Launch Oracle Database.

```Code
podman run -d --name 23aidb \
 -p 1521:1521 \
 -e ORACLE_SID=FREE \
 -e ORACLE_PDB=FREEPDB1 \
 -e ORACLE_PWD=Oracle4U \
 -e INIT_SGA_SIZE=2000 \
 -e INIT_PGA_SIZE=500 \
 -e ORACLE_EDITION=developer \
 -e ENABLE_ARCHIVELOG=false \
 -v OracleDBData:/opt/oracle/oradata \
container-registry.oracle.com/database/free
```

Launch ollama server.


```Code

podman run -d -v ollama:/home/ubuntu//.ollama -p 11434:11434 --name ollama docker.io/ollama/ollama

```

Download the model llama3.2

```Code
podman exec -it ollama ollama pull llama3.2
```

Check list of models.

```Code

podman exec -it ollama ollama list

NAME               ID              SIZE      MODIFIED       
llama3.2:latest    a80c4f17acd5    2.0 GB    38 seconds ago    

```



