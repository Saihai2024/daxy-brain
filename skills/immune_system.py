import re

class CognitiveImmuneSystem:
    def __init__(self):
        # 启发式规则：常见谣言/低置信度特征词
        self.suspicious_keywords = [
            "网传", "据说", "可能", "大概", "部分网友认为", 
            "谣言", "未证实", "或许", "传闻", "有人说", "道听途说"
        ]
        
    def evaluate_credibility(self, text):
        """
        Evaluate the credibility of a given text.
        Returns a float between 0.0 and 1.0
        """
        if not text or len(text) < 5:
            return 0.0
            
        score = 1.0
        
        # 1. 扫描低可信度词汇
        for kw in self.suspicious_keywords:
            if kw in text:
                score -= 0.15 # 发现一个可疑词扣除 0.15 分
                
        # 2. 标点符号异常检测 (例如过多的感叹号或问号暗示主观情绪强烈，不适合作为严谨知识)
        if text.count("！") + text.count("!") > 2:
            score -= 0.2
        if text.count("？") + text.count("?") > 1:
            score -= 0.2
            
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
