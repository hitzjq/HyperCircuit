import sys
from siflow import SiFlow

client = SiFlow(
    region="cn-beijing",    # 选定区域
    cluster="auriga",       # 选定区域对应的集群
    access_key_id="ebc64b15-9066-4e9e-b6a8-488d4dac6cd4",     # 从1.2页面上获取的 Access Key
    access_key_secret="Ptg1swcW1Jyo2m62f0"  # 从1.2页面上获取的 Secret Key
)

# 原来的部分：
# uuid = client.tasks.create(yaml_file="siflow.yml")
# print(uuid)

# 新的版本：从命令行读取 yml 文件，如果没有输入则默认使用 siflow.yml
if len(sys.argv) < 2:
    yaml_file = "siflow.yml"
else:
    yaml_file = sys.argv[1]

uuid = client.tasks.create(yaml_file=yaml_file)
print(f"[{yaml_file}] uuid: {uuid}")