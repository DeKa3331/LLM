# Running the Application

Follow the steps below to properly set up and run the application.


## 1. Download the project

Download the **entire folder** containing:

- Python source files
- JSON data files


## 2. Install requirements

Start by installing all required dependencies:

```bash
pip install -r requirements.txt
```


## 3. Configure environment variables

1. Locate the file: `.env.template`
2. Rename it to: `.env`
3. Open the file and fill in your API keys


## 4. Run the application

You can run the app in two ways:

### Option A — Classic run (it should run unicorn)

```bash
python api.py
```

### Option B — Using uvicorn (recommended i tested it this way)

```bash
uvicorn api:app --reload --port 8000
```


## 5. Test the API with Postman

I'm using Postman to send all the requests.

### Health Check

First, verify the API is running:

```
GET http://localhost:8000/health
```

![Health Check](https://github.com/user-attachments/assets/71367eef-9cc0-49c1-baef-42b94607f943)

### Ask Endpoint

Then use the `/ask` endpoint:

**Parameters:**
- **question**: Product you want to check micronutrients for
- **k**: Number of outputs (results)
- **mode**: Select model type (`local`, `gemini`, or `groq` if filled proper keys)
- **Content-Type**: Set to `application/json`

![Request Example 1](https://github.com/user-attachments/assets/257ae3cc-867b-459f-b76f-4c4a56a3d5b4)

![Request Example 2](https://github.com/user-attachments/assets/3ade9696-c7bd-436d-be4a-593ed5f04276)

![Request Example 3](https://github.com/user-attachments/assets/9598396a-b376-4a6b-81bd-a9e9bdc5c130)

### Important Notes

**Warning**: If you request too many responses, you will get an error.

![Timeout Example](https://github.com/user-attachments/assets/0bea5b2d-3c1b-44cd-aa90-a78724056630)

**Security**: Do not attempt to inject anything into the requests.

![SQL Injection Warning](https://github.com/user-attachments/assets/21c1425d-69fa-4694-8037-05882a40949d)




