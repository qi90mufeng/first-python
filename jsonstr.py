#!/usr/bin/python
import json


print("解码 JSON 对象")
jsonData = '{"a":1,"b":2,"c":3,"d":4,"e":5}';
text = json.loads(jsonData)
print(text)

print("编码成 JSON 字符串")
data = [{ 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}]
# 格式化输出
json = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '))
print(json)