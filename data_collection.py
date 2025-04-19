import requests


response = requests.get(url)

if response.status_code == 200:
    print("Request successful!")
    print(response.text) # Print the content of the response
    # or
    print(response.json()) # If the response is JSON, parse it
else:
    print(f"Request failed with status code {response.status_code}")