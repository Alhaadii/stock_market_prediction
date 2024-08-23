import os,json
path=os.path.dirname(os.path.abspath(__file__))+"\data.json"
print(path)
with open(path, 'r') as f:
    data=json.load(f)
Valid_Ticker =data['data']
print("Valid",Valid_Ticker)