import urllib.request
import urllib.parse
import re

def scrape_zhidao(question):
    try:
        url = 'https://zhidao.baidu.com/search?word=' + urllib.parse.quote(question)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        response = urllib.request.urlopen(req, timeout=5)
        html = response.read().decode('gbk', errors='ignore')
        
        # 匹配最佳答案摘要
        match = re.search(r'<dd class="dd answer">(.*?)</dd>', html, re.S)
        if match:
            ans = match.group(1)
            ans = re.sub(r'<.*?>', '', ans).strip()
            # 清理多余空格换行
            ans = re.sub(r'\s+', ' ', ans)
            if len(ans) > 10:
                return ans
    except Exception as e:
        print(f"Error: {e}")
    return None

print(scrape_zhidao("为什么天空是蓝色的"))
