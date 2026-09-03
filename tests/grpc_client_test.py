"""End-to-end smoke test for the cognitive gRPC service."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根

import time
import grpc
from chatbot_grpc import chatbot_pb2, chatbot_pb2_grpc


def main():
    channel = grpc.insecure_channel("localhost:9999")
    stub = chatbot_pb2_grpc.ChatBotServiceStub(channel)
    queries = [
        "光速",                # brain hit (web_corpus)
        "降维打击",            # brain hit
        "引力波",              # brain hit
        "宇宙",                # brain hit
        "你叫什么名字",        # brain associative hit -> 名字
        "量子纠缠是什么",      # brain weak match
        "黑洞",                # might miss brain -> fallback
        "我失恋了",            # brain miss -> brain Seq2Seq
        "晚安",                # brain miss -> brain
        "python好学吗",        # brain miss + qa -> recall fallback
    ]
    out = []
    for q in queries:
        req = chatbot_pb2.ReceivedMessage(
            user_id="u1", user_message=q, create_time=int(time.time())
        )
        resp = stub.Chatbot(req)
        line = f"Q: {q}\nA: {resp.user_response[:200]}\n"
        out.append(line)
    with open("_grpc_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("written _grpc_results.txt")


if __name__ == "__main__":
    main()
