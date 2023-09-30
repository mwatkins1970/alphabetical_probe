# GPT-J alphabetical probe
Experimental code which trains 26 linear probes to detect the presence of alphabetic letters in GPT-J token strings, given their embeddings. Exploring the resulting vector arithmetic and its impact on GPT-J spelling abilities


## Setup 

```bash
conda create --name alphabet python=3.11
conda activate alphabet
pip install -r requirements.txt
```