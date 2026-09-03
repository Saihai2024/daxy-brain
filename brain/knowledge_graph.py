import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import re
import json

class KnowledgeGraphEngine:
    def __init__(self):
        self.timeout = 5.0 

    def fetch_wikipedia_summary(self, concept):
        """Fetch the first snippet of a Wikipedia article for a concept."""
        try:
            url = f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(concept)}&utf8=&format=json"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=self.timeout)
            data = json.loads(response.read().decode('utf-8'))
            
            search_results = data.get('query', {}).get('search', [])
            if search_results:
                # Take the top result snippet and clean it
                snippet = search_results[0].get('snippet', '')
                soup = BeautifulSoup(snippet, "html.parser")
                text = soup.get_text()
                # Clean brackets and special chars
                text = re.sub(r'\[.*?\]', '', text)
                return text
        except Exception as e:
            pass
        return ""

    def infer_relationship(self, unknown_noun, known_concepts_list):
        """
        Attempt to map an unknown noun to a known concept via external knowledge retrieval.
        Returns the closest mapped known concept, or None if no link is found.
        """
        summary = self.fetch_wikipedia_summary(unknown_noun)
        if not summary:
            return None
            
        # Scan the summary to see if it mentions any concept the model already knows
        for known in known_concepts_list:
            # We want meaningful overlapping concepts, not single characters or extremely common stop words.
            # Assuming known_concepts_list are meaningful keys from concept_dict.
            if len(known) >= 2 and known in summary and known != unknown_noun:
                # Found a logic link! e.g., "火星" snippet contains "行星", and we know "行星".
                return known
                
        return None
