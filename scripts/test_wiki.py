import urllib.request
import urllib.parse
import json
import re

def scrape_zhidao(question):
    # Using Wikipedia Search API as a reliable fallback for questions
    try:
        url = 'https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch=' + urllib.parse.quote(question) + '&utf8=&format=json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5)
        data = json.loads(response.read().decode('utf-8'))
        
        search_results = data.get('query', {}).get('search', [])
        if search_results:
            snippet = search_results[0].get('snippet', '')
            # Clean HTML tags
            ans = re.sub(r'<.*?>', '', snippet).strip()
            # Clean multiple spaces
            ans = re.sub(r'\s+', ' ', ans)
            if len(ans) > 10:
                return ans
    except Exception as e:
        print(f"Error: {e}")
    return None

print(scrape_zhidao("为什么天空是蓝色的"))
