dowload whole folder with python files and json with data.
start with dowloading requirements
Install with: pip install -r requirements.txt
edit .env.temple, remove .temple leave only ".env" and fill with your keys.
next you can either run classic way api.py or use uvicorn api:app --reload --port 8000

from there you can check if it's working
http://localhost:8000/health
<img width="1494" height="780" alt="image" src="https://github.com/user-attachments/assets/71367eef-9cc0-49c1-baef-42b94607f943" />
then use ask
fill question with product you want to check mikros of
set 
Content-Type to "application/json"
k-stands fro number of outputs, 
you can also try different mode:
local/gemini/groq if you filled proper keys before

<img width="1473" height="1051" alt="image" src="https://github.com/user-attachments/assets/257ae3cc-867b-459f-b76f-4c4a56a3d5b4" />
<img width="1495" height="1032" alt="image" src="https://github.com/user-attachments/assets/3ade9696-c7bd-436d-be4a-593ed5f04276" />
<img width="1486" height="1038" alt="image" src="https://github.com/user-attachments/assets/9598396a-b376-4a6b-81bd-a9e9bdc5c130" />

if you try to get too many responses you will be timed out
<img width="1497" height="948" alt="image" src="https://github.com/user-attachments/assets/0bea5b2d-3c1b-44cd-aa90-a78724056630" />

don't try to inject anything
<img width="1479" height="758" alt="image" src="https://github.com/user-attachments/assets/21c1425d-69fa-4694-8037-05882a40949d" />




