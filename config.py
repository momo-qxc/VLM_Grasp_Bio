"""
全局配置文件 - 统一管理 API Keys、模型名称、URL 等
Global Configuration - Centralized management of API Keys, model names, URLs, etc.

使用方法 / Usage:
    from config import Config

    # 使用 Qwen API
    client = OpenAI(
        api_key=Config.QWEN_API_KEY,
        base_url=Config.QWEN_BASE_URL,
        http_client=httpx.Client(trust_env=False)
    )

    # 使用模型名称
    model = Config.QWEN_MODEL
"""


class Config:
    """全局配置类 - 所有配置项都在这里定义"""

    # ==================== API Keys ====================
    # 阿里云 Qwen VLM API Key
    QWEN_API_KEY = 'sk-3d29c129d0664685853e5311a2241127'

    # Google Gemini API Key
    GEMINI_API_KEY = 'AIzaSyBJvmAHO92kO4t0zKo7sqrtrnk9jmR3HRk'

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

    # ==================== 其他配置 ====================
    # 是否禁用代理（建议保持 True）
    DISABLE_PROXY = True

    @classmethod
    def get_qwen_client_config(cls):
        """获取 Qwen 客户端配置（返回字典，方便直接传参）"""
        return {
            'api_key': cls.QWEN_API_KEY,
            'base_url': cls.QWEN_BASE_URL
        }

    @classmethod
    def get_gemini_client_config(cls):
        """获取 Gemini 客户端配置"""
        return {
            'api_key': cls.GEMINI_API_KEY
        }

    @classmethod
    def validate(cls):
        """验证必要的配置是否已设置"""
        if not cls.QWEN_API_KEY or cls.QWEN_API_KEY == 'your_api_key_here':
            raise ValueError("请在 config.py 中设置 QWEN_API_KEY")
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == 'your_api_key_here':
            print("警告: GEMINI_API_KEY 未配置，Gemini 功能将不可用")
        return True


# 可选：在导入时自动验证配置
# Config.validate()
