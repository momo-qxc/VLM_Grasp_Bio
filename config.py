"""
全局配置文件 - 统一管理 API Keys、模型名称、URL 等
"""


class Config:
    # ==================== VLM 主模型（用于抓取识别）====================
    QWEN_API_KEY = 'API_KEY'
    QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    QWEN_MODEL = 'qwen-vl-max-latest'

    # ==================== 润色专用模型（不在 UI 设置中显示）====================
    POLISH_API_KEY = 'API_KEY'
    POLISH_BASE_URL = 'https://api.deepseek.com'
    POLISH_MODEL = 'deepseek-chat'

    # ==================== 用户可配置模型列表（UI 设置中管理）====================
    MODELS = {
        'qwen-vl-max-latest': {
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'key': 'API_KEY',
        },
        'gpt-5.2-pro-2025-12-11': {
            'url': 'https://api.openai.com/v1',
            'key': 'API_KEY',
        },
    }
    ACTIVE_MODEL = 'qwen-vl-max-latest'

    # ==================== 模型参数 ====================
    DEFAULT_TEMPERATURE = 0.1
    DISABLE_PROXY = True
    UI_FONT_SIZE = 14
    CAMERA_FPS = 10
    LOG_AUTOSCROLL = True

    # ==================== 机械臂工作空间 ====================
    ROBOT_BASE_X = 1.1
    ROBOT_BASE_Y = 0.3
    WORKSPACE_R_MIN = 0.15
    WORKSPACE_R_MAX = 0.82
    TABLE_X_MIN = 0.0
    TABLE_X_MAX = 1.6
    TABLE_Y_MIN = 0.0
    TABLE_Y_MAX = 1.2

    @classmethod
    def get_qwen_client_config(cls):
        return {'api_key': cls.QWEN_API_KEY, 'base_url': cls.QWEN_BASE_URL}

    @classmethod
    def create_qwen_client(cls):
        from openai import OpenAI
        import httpx
        return OpenAI(
            api_key=cls.QWEN_API_KEY,
            base_url=cls.QWEN_BASE_URL,
            http_client=httpx.Client(trust_env=False)
        )

    @classmethod
    def validate(cls):
        if not cls.QWEN_API_KEY or cls.QWEN_API_KEY == 'your_api_key_here':
            raise ValueError("请在 config.py 中设置 QWEN_API_KEY")
        return True
