# API 配置说明 / API Configuration Guide

## 简介 / Introduction

本项目使用 `config.py` 集中管理所有配置，包括 API Keys、模型名称、URL 等。

This project uses `config.py` for centralized configuration management, including API Keys, model names, URLs, etc.

---

## 快速开始 / Quick Start

### 1. 修改配置

所有配置都在 `config.py` 文件中，直接双击打开编辑即可：

```python
class Config:
    """全局配置类 - 所有配置项都在这里定义"""

    # ==================== API Keys ====================
    # 阿里云 Qwen VLM API Key
    QWEN_API_KEY = 'your_qwen_api_key_here'

    # Google Gemini API Key
    GEMINI_API_KEY = 'your_gemini_api_key_here'

    # ==================== API Base URLs ====================
    # 阿里云 Qwen API 地址
    QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

    # ==================== 模型名称 ====================
    # Qwen VLM 模型名称
    QWEN_MODEL = 'qwen-vl-max-latest'

    # Gemini 模型名称
    GEMINI_MODEL = 'gemini-robotics-er-1.5-preview'

    # ==================== 模型参数 ====================
    # 默认温度参数（控制输出随机性，越低越确定）
    DEFAULT_TEMPERATURE = 0.1
```

---

### 2. 更换平台

如需更换平台，只需修改 `config.py` 中的对应配置：

**示例 1：更换 API Key**
```python
QWEN_API_KEY = 'sk-new-api-key-here'
```

**示例 2：更换模型**
```python
QWEN_MODEL = 'qwen-vl-plus'  # 从 max 改为 plus
```

**示例 3：更换 API 地址**
```python
QWEN_BASE_URL = 'https://your-custom-api.com/v1'
```

---

### 3. 代码使用方式

代码会自动从 `config.py` 加载配置：

```python
from config import Config

# 使用 Qwen API
client = OpenAI(
    api_key=Config.QWEN_API_KEY,
    base_url=Config.QWEN_BASE_URL,
    http_client=httpx.Client(trust_env=False)
)

# 使用模型名称
completion = client.chat.completions.create(
    model=Config.QWEN_MODEL,
    messages=messages,
    temperature=Config.DEFAULT_TEMPERATURE
)

# 使用 Gemini API
client = genai.Client(api_key=Config.GEMINI_API_KEY)
```

---

## 优势 / Advantages

✅ **集中管理** - 所有配置在一个文件中，易于查找和修改
✅ **易于编辑** - 直接双击 `config.py` 用任何编辑器打开
✅ **易于切换** - 更换平台/模型只需修改一处
✅ **类型安全** - 使用类属性，IDE 可以自动补全
✅ **无需环境变量** - 不需要配置 .env 或系统环境变量

---

## 配置项说明 / Configuration Items

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `QWEN_API_KEY` | 阿里云 Qwen API 密钥 | 需要设置 |
| `QWEN_BASE_URL` | Qwen API 地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `QWEN_MODEL` | Qwen 模型名称 | `qwen-vl-max-latest` |
| `GEMINI_API_KEY` | Google Gemini API 密钥 | 需要设置 |
| `GEMINI_MODEL` | Gemini 模型名称 | `gemini-robotics-er-1.5-preview` |
| `DEFAULT_TEMPERATURE` | 模型温度参数 | `0.1` |

---

## 可用的模型 / Available Models

### Qwen VLM 模型
- `qwen-vl-max-latest` - 最新的 Max 版本（推荐）
- `qwen-vl-plus` - Plus 版本
- `qwen-vl-max` - Max 版本
- `qwen3-omni-flash` - Omni Flash 版本

### Gemini 模型
- `gemini-robotics-er-1.5-preview` - Robotics ER 1.5 预览版

---

## 注意事项 / Notes

1. **不要将 `config.py` 提交到公开仓库** - 如果需要分享代码，先删除 API Keys
2. **定期更换 API Keys** - 保证安全
3. **温度参数说明** - 越低（0.0-0.3）输出越确定，越高（0.7-1.0）输出越随机
4. **模型选择** - 根据任务需求选择合适的模型

---

## 故障排查 / Troubleshooting

### 问题：提示 API Key 未配置

**解决方案：**
1. 打开 `config.py` 文件
2. 找到 `QWEN_API_KEY` 或 `GEMINI_API_KEY`
3. 将 `'your_api_key_here'` 替换为实际的 API Key

### 问题：API 调用失败

**解决方案：**
1. 验证 API Key 是否有效
2. 检查 `QWEN_BASE_URL` 是否正确
3. 确认网络连接正常
4. 检查 API 服务是否正常运行

### 问题：想要切换模型

**解决方案：**
1. 打开 `config.py`
2. 修改 `QWEN_MODEL` 为其他可用模型
3. 保存文件，重新运行程序

---

## 高级用法 / Advanced Usage

### 使用辅助方法

`Config` 类提供了便捷方法：

```python
# 获取 Qwen 客户端配置
qwen_config = Config.get_qwen_client_config()
# 返回: {'api_key': '...', 'base_url': '...'}

# 获取 Gemini 客户端配置
gemini_config = Config.get_gemini_client_config()
# 返回: {'api_key': '...'}

# 验证配置
Config.validate()  # 检查必要的配置是否已设置
```

---

## 联系方式 / Contact

如有问题，请查看项目主 README 或提交 Issue。
