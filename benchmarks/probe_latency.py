import json, os, time, urllib.request, http.cookiejar, sys

BASE = os.environ.get("DASH_BASE", "http://127.0.0.1:8000")
USER = os.environ.get("DASH_USER", "root")
PASS = os.environ.get("DASH_PASS", "")
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

def post(path, obj, timeout=60):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with op.open(req, timeout=timeout) as r:
        body = r.read()
    return (time.perf_counter() - t0) * 1000, json.loads(body.decode("utf-8"))

ms, _ = post("/api/login", {"username": USER, "password": PASS})
print("login %.0fms" % ms)

QS = [
    "你好",
    "量子纠缠是什么",
    "python好学吗",
    "介绍一下上海",
    "什么是机器学习",
    "帮我把一个 FastAPI 服务的响应时间从 300ms 降到 100ms,应该先做什么",
    "asdkjhaskdjh",
    "你叫什么名字",
]
for q in QS:
    lat = []
    ans = ""
    for i in range(3):
        try:
            ms, j = post("/api/chat", {"message": q, "user_id": "probe"})
            lat.append(ms)
            ans = j.get("response", "")
        except Exception as e:
            print("ERR", q, e)
            break
    if lat:
        print("%7.0f %7.0f %7.0f  | %-28s | %s" % (
            lat[0], min(lat), max(lat), q[:28], ans[:60].replace("\n", " ")))
sys.stdout.flush()
