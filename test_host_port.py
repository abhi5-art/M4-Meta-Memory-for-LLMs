import requests
import time


BASE_URL = "https://corporate-economic-automatic.ngrok-free.dev"
# ========================================================

def test_health():
    print("\n--- 1. Testing Server Health ---")
    try:
        response = requests.get(BASE_URL)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text.strip()}")
    except Exception as e:
        print(f"Error: {e}")

def test_list_models():
    print("\n--- 2. Listing Available Models ---")
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"Found {len(models)} models:")
            for m in models:
                print(f" - {m['name']}")
        else:
            print(f"Failed with status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def test_chat_generation():
    print("\n--- 3. Testing Chat Generation (qwen2.5:7b-instruct) ---")
    payload = {
        "model": "qwen2.5:7b-instruct",
        "prompt": "Write a short haiku about APIs.",
        "stream": False
    }
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/generate", json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"Response Time: {elapsed:.2f} seconds")
            print("Output:")
            print(response.json().get('response', 'No response field found.'))
        else:
            print(f"Failed with status: {response.status_code}\n{response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_embeddings():
    print("\n--- 4. Testing Embeddings (all-minilm) ---")
    payload = {
        "model": "all-minilm",
        "prompt": "This is a test sentence to convert into dense vectors."
    }
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/embeddings", json=payload)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            embedding = response.json().get('embedding', [])
            print(f"Response Time: {elapsed:.2f} seconds")
            print(f"Embedding generated! Length of vector: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        else:
            print(f"Failed with status: {response.status_code}\n{response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if BASE_URL == "https://your-domain.ngrok-free.app":
        print("⚠️  Please update BASE_URL with your actual ngrok URL before running.")
    else:
        test_health()
        test_list_models()
        test_chat_generation()
        test_embeddings()