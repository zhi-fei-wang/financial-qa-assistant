# 部署指南

## 方式一：本地运行（比赛演示适用）

你在自己电脑上 `streamlit run app.py` 就是已经"部署"了。
评委可以看 `http://localhost:8501`。

如果评委在其他电脑上，同一局域网内：
```
http://你的IP地址:8501
```
查看 IP：终端输入 `ipconfig`，找 IPv4 地址。

---

## 方式二：局域网共享（最简单，无需注册任何平台）

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```
然后同一 WiFi 下的评委用浏览器打开 `http://你的IP:8501`。

---

## 方式三：公网部署（免费，评委在外网也能访问）

### Streamlit Cloud（推荐）

1. 把项目文件夹推送到 GitHub 仓库
2. 打开 https://share.streamlit.io → Sign in with GitHub
3. 点 "New app" → 选择你的仓库
4. Main file path: `app.py`
5. Advanced settings → Secrets: 添加 `DEEPSEEK_API_KEY`
6. 点 Deploy → 获得 `xxx.streamlit.app` 公开网址

**限制**：免费版资源有限（1GB RAM），首次启动构建图谱约 90 秒，后续热启动 ~10 秒。

### Hugging Face Spaces（备选）

1. 创建 `Dockerfile` 或直接使用 Streamlit SDK
2. 推到 Hugging Face Space
3. 设置 Secret: `DEEPSEEK_API_KEY`

---

## 方式四：云服务器（长期运行）

```bash
# 1. 登录云服务器
ssh user@your-server-ip

# 2. 安装 Python 3.10+ 和依赖
pip install -r requirements.txt

# 3. 上传项目文件（scp 或 git clone）
scp -r financial-qa-assistant/ user@server:/opt/

# 4. 设置环境变量
export DEEPSEEK_API_KEY=sk-xxx

# 5. 后台运行
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

# 6. 配置防火墙开放 8501 端口
```

---

## 数据说明

项目数据文件使用 gzip 压缩（`.csv.gz`）以适应 GitHub 100MB 限制。
`src/utils/data_loader.py` 已支持自动读取 `.csv.gz` 文件，无需手动解压。

若需要解压回原始 CSV（用于 Excel 查看等）：
```bash
python -c "
import gzip, shutil
files = ['2/clean.csv', '3/clean.csv', '4/asharebalancesheet_202605261517.csv', 
         '4/asharecashflow_202605261518.csv', '4/ashareincome_202605261519.csv',
         '5/rr_main_202605281537.csv']
for f in files:
    if os.path.exists(f + '.gz'):
        with gzip.open(f + '.gz', 'rb') as fin:
            with open(f, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        print(f'Decompressed: {f}')
"
```
