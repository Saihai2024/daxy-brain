import urllib.request
import urllib.parse
import re

def scrape_sogou(question):
    try:
        url = 'https://www.sogou.com/web?query=' + urllib.parse.quote(question)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        response = urllib.request.urlopen(req, timeout=5)
        html = response.read().decode('utf-8', errors='ignore')
        
        # 尝试匹配第一条摘要
        match = re.search(r'<div class="vr-summary[^>]*>(.*?)</div>', html, re.S)
        if not match:
            match = re.search(r'<div class="ft" id="[^>]*>(.*?)</div>', html, re.S)
        if not match:
            # Sogou Zhidao or general text
            match = re.search(r'<div class="space-txt">(.*?)</div>', html, re.S)
            
        if match:
            ans = match.group(1)
            ans = re.sub(r'<.*?>', '', ans).strip()
            ans = re.sub(r'\s+', ' ', ans)
            if len(ans) > 10:
                return ans
    except Exception as e:
        print(f"Error: {e}")
    return None

print(scrape_sogou("为什么天空是蓝色的"))
