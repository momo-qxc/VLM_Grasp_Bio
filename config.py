"""
全局配置文件 - 统一管理 API Keys、模型名称、URL 等
"""


class Config:
    # ==================== 润色专用模型（不在 UI 设置中显示）====================
    POLISH_API_KEY = 'your_key'
    POLISH_BASE_URL = 'https://api.deepseek.com'
    POLISH_MODEL = 'deepseek-chat'

    # ==================== 用户可配置模型列表（UI 设置中管理）====================
    MODELS = {
        'qwen-vl-max-latest': {
            'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'key': 'your_key',
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
        """获取当前激活模型的配置"""
        active_config = cls.MODELS.get(cls.ACTIVE_MODEL)
        if not active_config:
            raise ValueError(f"未找到模型配置: {cls.ACTIVE_MODEL}")
        return {
            'api_key': active_config.get('key', ''),
            'base_url': active_config.get('url', '')
        }

    @classmethod
    def create_qwen_client(cls):
        """创建使用当前激活模型配置的客户端"""
        from openai import OpenAI
        import httpx
        config = cls.get_qwen_client_config()
        return OpenAI(
            api_key=config['api_key'],
            base_url=config['base_url'],
            http_client=httpx.Client(trust_env=False)
        )

    @classmethod
    def get_active_model_name(cls):
        """获取当前激活的模型名称"""
        return cls.ACTIVE_MODEL

    @classmethod
    def validate(cls):
        """验证当前激活模型的配置是否有效"""
        if not cls.ACTIVE_MODEL:
            raise ValueError("未设置 ACTIVE_MODEL")
        if cls.ACTIVE_MODEL not in cls.MODELS:
            raise ValueError(f"ACTIVE_MODEL '{cls.ACTIVE_MODEL}' 不在 MODELS 列表中")
        config = cls.MODELS[cls.ACTIVE_MODEL]
        if not config.get('key'):
            raise ValueError(f"模型 '{cls.ACTIVE_MODEL}' 缺少 API Key")
        if not config.get('url'):
            raise ValueError(f"模型 '{cls.ACTIVE_MODEL}' 缺少 API URL")
        return True
