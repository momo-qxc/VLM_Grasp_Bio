"""
VLM 机器人抓取 - PyQt5 图形界面控制台
运行方式: conda activate vlm_graspnet_RRT && python ui_main_qt.py
"""
import os, sys

# ── 必须在所有其他 import 之前：修正 LD_LIBRARY_PATH 后重启自身 ──────────────
# 原因：conda base 的 /home/robot/anaconda3/lib/ 含 Qt 5.15.2，
#       PyQt5 是 Qt 5.15.18，两者共存会触发 "Cannot mix incompatible Qt library"。
# LD_LIBRARY_PATH 修改只对子进程生效，所以用 os.execv 重启来继承新路径。
_REEXEC_FLAG = "_PYQT5_LD_FIXED"
if not os.environ.get(_REEXEC_FLAG):
    try:
        import PyQt5 as _p
        _qt_lib  = os.path.join(os.path.dirname(_p.__file__), "Qt5", "lib")
        _qt_plug = os.path.join(os.path.dirname(_p.__file__), "Qt5", "plugins")
        _old_ld  = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _qt_lib + (":" + _old_ld if _old_ld else "")
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_plug
        os.environ[_REEXEC_FLAG] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"[UI] LD_LIBRARY_PATH 设置失败，继续尝试: {e}")

import threading, queue, time, io, math

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QScrollArea,
    QFrame, QRadioButton, QButtonGroup, QDialog, QDialogButtonBox,
    QSizePolicy, QComboBox, QSplitter, QGraphicsDropShadowEffect, QListWidget, QSpinBox, QSlider
)
from PyQt5.QtCore import (
    Qt, QTimer, QObject, pyqtSignal, QThread, QSize, QPointF,
    QPropertyAnimation, QEasingCurve, QRect, QSequentialAnimationGroup
)
from PyQt5.QtGui import (
    QFont, QPixmap, QImage, QColor, QPalette, QFontDatabase,
    QPainter, QPolygonF, QPainterPath, QPen
)

# ── 颜色常量 ──────────────────────────────────────────────
BG_DARK    = "#0a0a14"
BG_PANEL   = "#0f111a"
BG_CARD    = "#151822"
BG_INPUT   = "#1a1e2e"
BORDER_CLR = "#1e2235"
CYAN       = "#22d3ee"
CYAN_DIM   = "#164e63"
INDIGO     = "#818cf8"
INDIGO_DIM = "#312e81"
EMERALD    = "#34d399"
AMBER      = "#fbbf24"
ROSE       = "#fb7185"
TEXT_PRI   = "#e2e8f0"
TEXT_SEC   = "#64748b"
TEXT_DIM   = "#475569"


class CustomDropdown(QWidget):
    """
    完全自定义的下拉框组件，从根本上杜绝 QComboBox 原生系统的白边和虚线
    """
    currentTextChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self._current_text = ""

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 1. 触发按钮（伪装成 ComboBox 的显示区域）
        self.btn = QPushButton()
        self.btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_INPUT};
                color: {TEXT_PRI};
                border: 1px solid {BORDER_CLR};
                border-radius: 8px;
                padding: 5px 10px;
                text-align: left;
                font-size: 16px;
                min-height: 32px;
                outline: none;
            }}
            QPushButton:hover {{
                border: 1px solid {CYAN};
            }}
        """)
        self.btn.clicked.connect(self.show_popup)
        self.layout.addWidget(self.btn)

        # 2. 独立的下拉弹窗（使用 Qt.Popup 确保点击外部自动关闭）
        self.popup = QWidget(self, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.popup.setAttribute(Qt.WA_TranslucentBackground) # 弹窗底层透明，杜绝直角白底
        
        self.popup_layout = QVBoxLayout(self.popup)
        self.popup_layout.setContentsMargins(0, 0, 0, 0)
        self.popup_layout.setSpacing(0)

        # 3. 弹窗里的列表控件
        self.list_widget = QListWidget()
        self.list_widget.setFocusPolicy(Qt.NoFocus)  # 彻底杀死焦点虚线框
        self.list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {BG_CARD};
                color: {TEXT_PRI};
                border: 1px solid {BORDER_CLR};
                border-radius: 8px;
                outline: 0;
                padding: 4px;
            }}
            QListWidget::item {{
                border-radius: 4px;
                padding: 8px;
                color: {TEXT_PRI};
                outline: none;
            }}
            QListWidget::item:hover, QListWidget::item:selected {{
                background-color: {CYAN_DIM}; /* 悬浮颜色 */
                color: white;
                outline: none;
            }}
        """)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        
        self.popup_layout.addWidget(self.list_widget)

    def addItem(self, text):
        self.items.append(text)
        self.list_widget.addItem(text)
        if len(self.items) == 1:
            self._current_text = text
            self.btn.setText(text)

    def removeItem(self, index):
        if 0 <= index < len(self.items):
            self.items.pop(index)
            item = self.list_widget.takeItem(index)
            if item:
                del item
            if self.items:
                if self.currentIndex() == -1:
                    self.setCurrentIndex(max(0, index - 1))
            else:
                self._current_text = ""
                self.btn.setText("")
                self.currentTextChanged.emit("")

    def currentText(self):
        return self._current_text

    def findText(self, text):
        try:
            return self.items.index(text)
        except ValueError:
            return -1

    def setCurrentIndex(self, idx):
        if 0 <= idx < len(self.items):
            self._current_text = self.items[idx]
            self.btn.setText(self._current_text)
            self.currentTextChanged.emit(self._current_text)

    def setCurrentText(self, text):
        idx = self.findText(text)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def setItemText(self, idx, text):
        if 0 <= idx < len(self.items):
            old_text = self.items[idx]
            self.items[idx] = text
            item = self.list_widget.item(idx)
            if item:
                item.setText(text)
            if self._current_text == old_text:
                self._current_text = text
                self.btn.setText(text)

    def currentIndex(self):
        try:
            return self.items.index(self._current_text)
        except ValueError:
            return -1

    def show_popup(self):
        if not self.items: return
        
        # 动态计算弹窗高度
        item_height = 36
        total_height = len(self.items) * item_height + 10
        self.popup.resize(self.btn.width(), total_height)

        # 获取按钮在屏幕上的全局坐标，将弹窗显示在按钮正下方
        pos = self.btn.mapToGlobal(QPointF(0, self.btn.height() + 4).toPoint())
        self.popup.move(pos)
        self.popup.show()

    def on_item_clicked(self, item):
        self._current_text = item.text()
        self.btn.setText(self._current_text)
        self.popup.hide()
        self.currentTextChanged.emit(self._current_text)

# ── 触感增强按钮基类 ────────────────────────────────────────
class TactileButton(QPushButton):
    """具有缩放和阴影反馈的触感增强按钮"""
    def __init__(self, text="", parent=None, hover_color=CYAN):
        super().__init__(text, parent)
        self.setText(text)
        self._hover_color = hover_color
        self._is_pressed = False
        
        # 阴影效果
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(15)
        self._shadow.setColor(QColor(0, 0, 0, 150))
        self._shadow.setOffset(0, 2)
        self._shadow.setEnabled(False)
        self.setGraphicsEffect(self._shadow)

    def enterEvent(self, event):
        self._shadow.setEnabled(True)
        self._shadow.setColor(QColor(self._hover_color))
        self._shadow.setBlurRadius(20)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._shadow.setEnabled(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._is_pressed = True
        self._shadow.setOffset(0, 0)
        self._shadow.setBlurRadius(10)
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._is_pressed = False
        self._shadow.setOffset(0, 2)
        self._shadow.setBlurRadius(20)
        self.update()
        super().mouseReleaseEvent(event)

class ActionButton(TactileButton):
    """专用大操作按钮，带渐变色和强触感"""
    def __init__(self, text, parent=None, color=CYAN, dim_color=CYAN_DIM):
        super().__init__(text, parent, hover_color=color)
        self._color = color
        self._dim_color = dim_color

# ── 自定义发送按钮（向上箭头）────────────────────────────
class SendButton(TactileButton):
    """绘制向上箭头发送图标，增强触感"""
    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setObjectName("btn_send")
        self.setFixedSize(42, 42)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # 背景
        bg_color = QColor("#0e7490") if self._is_pressed else (QColor("#155e75") if self.underMouse() else QColor(CYAN_DIM))
        p.setBrush(bg_color)
        p.setPen(QPen(QColor(CYAN), 1) if self.underMouse() else Qt.NoPen)
        p.drawRoundedRect(self.rect().adjusted(1,1,-1,-1), 10, 10)

        # 偏移量模拟按下感
        off = 2 if self._is_pressed else 0
        cx, cy = w / 2.0, h / 2.0 + off
        
        # 颜色
        white = QColor("#ffffff")
        if not self.isEnabled():
            white = QColor(TEXT_DIM)
            
        # 箭头杆（竖线）
        pen = QPen(white, 3.0, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(cx, cy + h * 0.22), QPointF(cx, cy - h * 0.18))
        
        # 箭头头部（两条斜线）
        tip_y = cy - h * 0.18
        p.drawLine(QPointF(cx, tip_y), QPointF(cx - w * 0.18, tip_y + h * 0.16))
        p.drawLine(QPointF(cx, tip_y), QPointF(cx + w * 0.18, tip_y + h * 0.16))
        p.end()

# ── 自定义齿轮设置按钮 ────────────────────────────────────
class SettingsButton(QPushButton):
    """绘制齿轮图标的设置按钮，避免 emoji 在 Qt/Linux 上被裁剪"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none; border-radius: 8px; }}
            QPushButton:hover {{ background: {BG_CARD}; }}
            QPushButton:pressed {{ background: {BG_INPUT}; }}
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0 + (1 if self.isDown() else 0)
        color = QColor(CYAN) if self.underMouse() else QColor(TEXT_SEC)
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        R  = min(w, h) * 0.36   # 齿顶圆半径
        ri = min(w, h) * 0.25   # 齿根圆半径
        rc = min(w, h) * 0.12   # 中心孔半径
        n  = 8                   # 齿数
        tw = math.pi / n * 0.55  # 每齿半角宽
        gear = QPainterPath()
        for i in range(n):
            base = 2 * math.pi * i / n
            a1, a2 = base - tw, base + tw
            a3 = base + tw + (math.pi / n - tw)
            a4 = base + 2 * math.pi / n - tw - (math.pi / n - tw)
            if i == 0:
                gear.moveTo(cx + R * math.cos(a1), cy + R * math.sin(a1))
            else:
                gear.lineTo(cx + R * math.cos(a1), cy + R * math.sin(a1))
            gear.lineTo(cx + R  * math.cos(a2), cy + R  * math.sin(a2))
            gear.lineTo(cx + ri * math.cos(a3), cy + ri * math.sin(a3))
            gear.lineTo(cx + ri * math.cos(a4), cy + ri * math.sin(a4))
        gear.closeSubpath()
        hole = QPainterPath()
        hole.addEllipse(QPointF(cx, cy), rc, rc)
        p.drawPath(gear.subtracted(hole))
        p.end()

class CamLabel(QLabel):
    """相机显示标签：用 paintEvent 绘制图像，完全避免 setPixmap 触发 updateGeometry 导致布局跳变"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cam_pixmap = None

    def set_frame(self, pixmap):
        """替代 setPixmap，不触发布局重算"""
        self._cam_pixmap = pixmap
        self.update()  # 只重绘，不重算布局

    def paintEvent(self, event):
        if self._cam_pixmap is None:
            super().paintEvent(event)
            return
        p = QPainter(self)
        scaled = self._cam_pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width()  - scaled.width())  // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        p.end()

    def sizeHint(self):
        return QSize(640, 640)
    def minimumSizeHint(self):
        return QSize(320, 320)


class ChatInputEdit(QTextEdit):
    """支持自动变高、Shift+Enter换行、Enter发送的圆角输入框"""
    send_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("输入自然语言指令...")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAcceptRichText(False)
        self.setTabChangesFocus(False)
        
        # 优化内部边距，文字居中且不被切断
        self.document().setDocumentMargin(12)
        
        # 初始高度（一行）
        self.setFixedHeight(50)
        self.textChanged.connect(self._adjust_height)

    def _adjust_height(self):
        # 强制更新文档宽度以计算高度
        self.document().setTextWidth(self.viewport().width())
        doc_height = self.document().size().height()
        
        # 限制在 50px (1行) 到 160px (约5行) 之间
        max_h = 160
        new_height = max(50, min(max_h, int(doc_height) + 4))
        
        if new_height != self.height():
            self.setFixedHeight(new_height)
        
        # 只有真正超过最大高度时才显示滚动条并取消隐藏样式
        if doc_height > max_h - 10:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.verticalScrollBar().setStyleSheet("")
        else:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.verticalScrollBar().setStyleSheet("width: 0px; background: transparent;")

    def keyPressEvent(self, event):
        # Enter 发送，Shift+Enter 换行
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
        else:
            super().keyPressEvent(event)

# ── Toast 浮动通知 ────────────────────────────────────────
class Toast(QLabel):
    """短暂显示后渐隐的浮动通知"""
    def __init__(self, parent, text, success=True):
        super().__init__(text, parent)
        bg = "#064e3b" if success else "#4c0519"
        border = EMERALD if success else ROSE
        self.setStyleSheet(f"""
            QLabel {{ background: {bg}; color: {"#6ee7b7" if success else "#fda4af"};
                border: 1px solid {border}; border-radius: 8px;
                padding: 8px 18px; font-size: 13px; }}
        """)
        self.adjustSize()
        pw, ph = parent.width(), parent.height()
        self.move((pw - self.width()) // 2, ph - 80)
        self.raise_()
        self.show()
        self._opacity = 1.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._fade)
        self._timer.start(40)
        self._delay = 0

    def _fade(self):
        self._delay += 1
        if self._delay < 40:   # 等待约 1.6s 再开始渐隐
            return
        self._opacity -= 0.05
        if self._opacity <= 0:
            self._timer.stop()
            self.deleteLater()
            return
        self.setWindowOpacity(self._opacity)
        # QLabel 不支持 setWindowOpacity，用 stylesheet alpha 模拟
        alpha = int(self._opacity * 255)
        bg = f"rgba(6,78,59,{alpha})" if "6ee7b7" in self.styleSheet() else f"rgba(76,5,25,{alpha})"
        clr = f"rgba(110,231,183,{alpha})" if "6ee7b7" in self.styleSheet() else f"rgba(253,164,175,{alpha})"
        self.setStyleSheet(f"""
            QLabel {{ background: {bg}; color: {clr};
                border: 1px solid rgba(52,211,153,{alpha}); border-radius: 8px;
                padding: 8px 18px; font-size: 13px; }}
        """)
log_queue    = queue.Queue()
result_queue = queue.Queue()

# ── 重定向 stdout/stderr ──────────────────────────────────
class QueueWriter(io.TextIOBase):
    def __init__(self, q, tag="INFO"):
        self.q, self.tag = q, tag
    def write(self, text):
        if text.strip():
            self.q.put((self.tag, text.rstrip()))
        return len(text)
    def flush(self): pass

# ── 全局样式表（仅针对聊天区域进行字体缩放）────────────────────────
def build_stylesheet(chat_fs=16):
    ui_fs = 13  # 固定其余 UI 元素的字号
    
    return f"""
QMainWindow, QWidget {{ background: {BG_DARK}; color: {TEXT_PRI}; }}
QDialog {{ background: {BG_DARK}; color: {TEXT_PRI}; }}
QLabel {{ color: {TEXT_PRI}; background: transparent; font-size: {ui_fs}px; }}
QPushButton {{
    background: {BG_CARD}; color: {TEXT_PRI};
    border: 1px solid {BORDER_CLR}; border-radius: 8px;
    padding: 6px 14px; font-size: {ui_fs + 1}px;
}}
/* 针对聊天气泡和输入框使用动态字号 */
QLabel#chat_bubble_user, QLabel#chat_bubble_ai, ChatInputEdit {{
    font-size: {chat_fs}px;
}}
QLineEdit {{
    background: {BG_INPUT}; color: {TEXT_PRI};
    border: 1px solid {BORDER_CLR}; border-radius: 20px;
    padding: 6px 16px; font-size: {ui_fs}px;
}}
QTextEdit {{
    background: {BG_INPUT}; color: {TEXT_PRI};
    border: 1px solid {BORDER_CLR}; border-radius: 22px;
    padding: 0px 16px;
}}
QPushButton:hover {{ 
    background: {BG_INPUT}; 
    border-color: {CYAN}; 
    color: white;
}}
QPushButton:pressed {{
    background: {BG_DARK};
    padding-top: 8px;
    padding-left: 16px;
}}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER_CLR}; background: {BG_DARK}; }}

QPushButton#btn_exec {{
    background: {CYAN_DIM}; color: {CYAN};
    border: 1px solid {CYAN_DIM}; font-weight: bold;
    border-radius: 10px;
}}
QPushButton#btn_exec:hover {{ background: #155e75; border-color: {CYAN}; color: white; }}
QPushButton#btn_exec:pressed {{ background: #0e7490; padding-top: 10px; }}

QPushButton#btn_stop {{
    background: {BG_INPUT}; color: {TEXT_SEC};
    border: 1px solid {BORDER_CLR}; border-radius: 10px;
}}
QPushButton#btn_stop:hover {{ background: #3f1219; color: {ROSE}; border-color: {ROSE}; }}
QPushButton#btn_stop:pressed {{ background: #5a1a24; padding-top: 10px; }}

QPushButton#btn_sparkle {{
    background: {INDIGO_DIM}; color: {INDIGO};
    border: 1px solid #4338ca; border-radius: 10px;
}}
QPushButton#btn_sparkle:hover {{ background: #3730a3; border-color: {INDIGO}; color: white; }}
QPushButton#btn_sparkle:pressed {{ background: #4338ca; padding-top: 2px; }}

QLineEdit:focus, QTextEdit:focus {{ border-color: {CYAN_DIM}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {BG_PANEL}; width: 6px; border-radius: 3px; }}
QScrollBar::handle:vertical {{ background: {BORDER_CLR}; border-radius: 3px; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QComboBox {{
    background: {BG_INPUT}; color: {TEXT_PRI};
    border: 1px solid {BORDER_CLR}; border-radius: 8px;
    padding: 6px 10px; font-size: {ui_fs}px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: {BG_CARD}; color: {TEXT_PRI};
    border: 1px solid {BORDER_CLR};
    selection-background-color: {CYAN_DIM};
    outline: none; font-size: {ui_fs}px; padding: 2px;
}}
QRadioButton {{ color: {TEXT_PRI}; spacing: 8px; font-size: {ui_fs}px; }}
QRadioButton::indicator {{
    width: 14px; height: 14px; border-radius: 7px;
    border: 2px solid {TEXT_SEC}; background: transparent;
}}
QRadioButton::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QSplitter::handle {{ background: {BG_DARK}; border: none; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical {{ height: 4px; }}
QSplitter::handle:hover {{ background: {CYAN_DIM}; }}
"""


# ── 线程信号（线程安全 UI 更新）────────────────────────────
class UISignals(QObject):
    add_chat   = pyqtSignal(str, str)   # text, side
    add_log    = pyqtSignal(str, str)   # tag, msg
    update_cam = pyqtSignal(str, object) # key, bgr_img
    task_done  = pyqtSignal()
    set_status = pyqtSignal(str)        # ready|running|error
    plan_done  = pyqtSignal(bool, str)  # ok, result

# ── 设置读写（从 config.py 读取，保存也写回 config.py）──────
def load_settings():
    try:
        from config import Config
        models = {k: dict(v) for k, v in Config.MODELS.items()}
        active = Config.ACTIVE_MODEL
        font_size = getattr(Config, 'UI_FONT_SIZE', 13)
    except Exception:
        models = {"qwen-vl-max-latest": {
            "url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key": ""}}
        active = "qwen-vl-max-latest"
        font_size = 13
    if active not in models:
        active = next(iter(models), "")
    m = models.get(active, {})
    return {
        "models": models, "current_model": active,
        "api_url": m.get("url", ""), "api_key": m.get("key", ""),
        "model_name": active, "font_size": font_size,
    }

def save_to_config(models: dict, active_model: str, font_size: int = 13):
    """将模型列表和当前选择写回 config.py，其余配置保持不变"""
    config_path = os.path.join(ROOT_DIR, "config.py")
    try:
        from config import Config
        # 构建 MODELS 字符串
        m_lines = ["{\n"]
        for name, info in models.items():
            m_lines.append(f"        {repr(name)}: {{\n")
            m_lines.append(f"            'url': {repr(info.get('url', ''))},\n")
            m_lines.append(f"            'key': {repr(info.get('key', ''))},\n")
            m_lines.append(f"        }},\n")
        m_lines.append("    }")
        models_repr = "".join(m_lines)

        polish_key  = repr(getattr(Config, 'POLISH_API_KEY',  'sk-f668b6a0a68643dea174b74e30ecf9b1'))
        polish_url  = repr(getattr(Config, 'POLISH_BASE_URL', 'https://api.deepseek.com'))
        polish_model= repr(getattr(Config, 'POLISH_MODEL',    'deepseek-chat'))

        content = f'''"""
全局配置文件 - 统一管理 API Keys、模型名称、URL 等
"""


class Config:
    # ==================== VLM 主模型（用于抓取识别）====================
    QWEN_API_KEY = {repr(Config.QWEN_API_KEY)}
    QWEN_BASE_URL = {repr(Config.QWEN_BASE_URL)}
    QWEN_MODEL = {repr(Config.QWEN_MODEL)}

    # ==================== 润色专用模型（不在 UI 设置中显示）====================
    POLISH_API_KEY = {polish_key}
    POLISH_BASE_URL = {polish_url}
    POLISH_MODEL = {polish_model}

    # ==================== 用户可配置模型列表（UI 设置中管理）====================
    MODELS = {models_repr}
    ACTIVE_MODEL = {repr(active_model)}

    # ==================== 模型参数 ====================
    DEFAULT_TEMPERATURE = {getattr(Config, 'DEFAULT_TEMPERATURE', 0.1)}
    DISABLE_PROXY = {getattr(Config, 'DISABLE_PROXY', True)}
    UI_FONT_SIZE = {font_size}
    CAMERA_FPS = {getattr(Config, 'CAMERA_FPS', 15)}
    LOG_AUTOSCROLL = {getattr(Config, 'LOG_AUTOSCROLL', True)}

    # ==================== 机械臂工作空间 ====================
    ROBOT_BASE_X = {getattr(Config, 'ROBOT_BASE_X', 1.1)}
    ROBOT_BASE_Y = {getattr(Config, 'ROBOT_BASE_Y', 0.3)}
    WORKSPACE_R_MIN = {getattr(Config, 'WORKSPACE_R_MIN', 0.15)}
    WORKSPACE_R_MAX = {getattr(Config, 'WORKSPACE_R_MAX', 0.82)}
    TABLE_X_MIN = {getattr(Config, 'TABLE_X_MIN', 0.0)}
    TABLE_X_MAX = {getattr(Config, 'TABLE_X_MAX', 1.6)}
    TABLE_Y_MIN = {getattr(Config, 'TABLE_Y_MIN', 0.0)}
    TABLE_Y_MAX = {getattr(Config, 'TABLE_Y_MAX', 1.2)}

    @classmethod
    def get_qwen_client_config(cls):
        return {{'api_key': cls.QWEN_API_KEY, 'base_url': cls.QWEN_BASE_URL}}

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
'''
        with open(config_path, "w") as f:
            f.write(content)
        import importlib, sys as _sys
        if 'config' in _sys.modules:
            importlib.reload(_sys.modules['config'])
    except Exception as e:
        print(f"[save_to_config] 失败: {e}", flush=True)
        raise

# ── 辅助：创建分隔线 ──────────────────────────────────────
def make_separator(horizontal=True):
    line = QFrame()
    line.setFrameShape(QFrame.HLine if horizontal else QFrame.VLine)
    line.setStyleSheet(f"color: {BORDER_CLR}; background: {BORDER_CLR};")
    line.setFixedHeight(1) if horizontal else line.setFixedWidth(1)
    return line

# ── 辅助：创建卡片容器 ────────────────────────────────────
def make_card(parent=None):
    w = QWidget(parent)
    w.setStyleSheet(f"""
        QWidget {{
            background: {BG_CARD};
            border: 1px solid {BORDER_CLR};
            border-radius: 10px;
        }}
    """)
    return w

# ── 辅助：节标题 Label ────────────────────────────────────
def make_section_label(text, parent=None):
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 19px; font-weight: bold;"
                      f" background: transparent; border: none;")
    return lbl

# ══════════════════════════════════════════════════════════
#  主窗口
# ══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("机器人智能抓取平台")
        self.resize(1720, 1000)
        self.setMinimumSize(1280, 800)

        # ── 状态 ──
        self.status      = "ready"
        self.running     = False
        self.env         = None
        self.env_ready   = False
        self._task_queue = queue.Queue()
        self._log_buffer = []
        self._chat_mode  = "normal"
        self._clarification_event  = threading.Event()
        self._clarification_result = None
        self._settings   = load_settings()
        self.is_planning  = False
        self.is_diagnosing = False

        # ── 信号 ──
        self.sig = UISignals()
        self.sig.add_chat.connect(self._add_chat_bubble)
        self.sig.add_log.connect(self._append_log)
        self.sig.update_cam.connect(self._update_camera)
        self.sig.task_done.connect(self._on_task_done)
        self.sig.set_status.connect(self._set_status)
        self.sig.plan_done.connect(self._on_plan_done)

        self._build_ui()

        # 轮询定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_queues)
        self._timer.start(50)

        # 欢迎消息
        self._add_chat_bubble("机器人智能抓取平台已启动。", "system")
        self._add_chat_bubble(
            "您好！我是智能视觉抓取助手。您可以直接下发操作指令，"
            "或者输入初步想法并点击 ✨ 按钮，我会为您将其转化为精准的机器人执行命令。", "ai")

        # 启动 MuJoCo 线程
        threading.Thread(target=self._mujoco_loop, daemon=True).start()

    # ══════════════════════════════════════════════════════
    #  构建 UI
    # ══════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        # ── 1. 创建左右中的三个面板 ──
        left_p   = self._build_left_panel()
        center_p = self._build_center_panel()
        right_p  = self._build_right_panel()

        # ── 2. 设置最小宽度，防止被挤没 ──
        left_p.setMinimumWidth(220)
        center_p.setMinimumWidth(500)
        right_p.setMinimumWidth(280)

        # ── 3. 使用 QSplitter 实现主布局 ──
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.addWidget(left_p)
        self.main_splitter.addWidget(center_p)
        self.main_splitter.addWidget(right_p)
        
        # ── 4. 禁止折叠（最重要：防止拖动时消失） ──
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, False)
        
        # 设置伸缩权重，中间区域最灵活
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        
        # 初始尺寸分配
        self.main_splitter.setSizes([250, 1220, 250])
        
        # 延迟再次设置尺寸，确保在窗口显示后布局生效，解决“初始窄”的问题
        QTimer.singleShot(200, lambda: self.main_splitter.setSizes([250, 1220, 250]))

        body_wrapper = QWidget()
        body_lay = QVBoxLayout(body_wrapper)
        body_lay.setContentsMargins(8, 6, 8, 8)
        body_lay.addWidget(self.main_splitter)
        
        root_layout.addWidget(body_wrapper, 1)

    def _build_header(self):
        hdr = QWidget()
        hdr.setFixedHeight(50)
        hdr.setStyleSheet(f"background: {BG_PANEL}; border-bottom: 1px solid {BORDER_CLR};")
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(16, 0, 16, 0)

        logo = QLabel("🤖")
        logo.setStyleSheet(f"color: {CYAN}; font-size: 20px; font-weight: bold;")
        title = QLabel("机器人智能抓取平台")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        lay.addWidget(logo)
        lay.addWidget(title)
        lay.addStretch()

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {EMERALD}; font-size: 20px;")
        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet(f"color: {EMERALD}; font-size: 20px;")
        lay.addWidget(self.status_dot)
        lay.addWidget(self.status_label)
        lay.addSpacing(16)

        btn_settings = SettingsButton()
        btn_settings.clicked.connect(self._open_settings)
        lay.addWidget(btn_settings)
        return hdr

    def _build_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(220)
        panel.setStyleSheet(f"background: {BG_PANEL}; border: 1px solid {BORDER_CLR}; border-radius: 12px;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # 快捷控制
        ctrl = make_card()
        ctrl_lay = QVBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(10, 10, 10, 10)
        ctrl_lay.addWidget(make_section_label("⚡ 快捷控制"))
        
        self.btn_exec = ActionButton("▶️  执行任务", color=CYAN, dim_color=CYAN_DIM)
        self.btn_exec.setObjectName("btn_exec")
        self.btn_exec.setFixedHeight(38)
        self.btn_exec.clicked.connect(self._on_execute)
        
        self.btn_stop = ActionButton("⏹️  紧急停止", color=ROSE, dim_color=BG_INPUT)
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        
        ctrl_lay.addWidget(self.btn_exec)
        ctrl_lay.addWidget(self.btn_stop)
        lay.addWidget(ctrl)

        # 快捷指令
        ex_card = make_card()
        ex_lay = QVBoxLayout(ex_card)
        ex_lay.setContentsMargins(10, 10, 10, 10)
        ex_lay.addWidget(make_section_label("💻 快捷指令"))
        examples = ["把培养皿放到显微镜右边",
                    "把培养皿放回货架",
                    "把培养皿放到显微镜左边一点",
                    "把培养皿放到绿色区域左边5厘米",
                    "把培养皿放回上次的位置",
                    "把鸭子放到显微镜左边"]
        for ex in examples:
            btn = QPushButton(ex)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {BG_DARK}; color: {TEXT_PRI};
                    border: 1px solid {BORDER_CLR}; border-radius: 8px;
                    padding: 6px 10px; font-size: 18px; text-align: left;
                }}
                QPushButton:hover {{
                    background: {CYAN_DIM}; border-color: {CYAN_DIM};
                    color: {CYAN};
                }}
            """)
            btn.clicked.connect(lambda _, t=ex: self._fill_input(t))
            ex_lay.addWidget(btn)

        ex_lay.addStretch(1)
        ex_lay.addWidget(make_separator())
        btn_log = QPushButton("📋  查看底层调试日志")
        btn_log.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_SEC};
                border: 1px solid {BORDER_CLR}; border-radius: 8px;
                padding: 8px; font-size: 18px;
            }}
            QPushButton:hover {{
                background: {CYAN_DIM}; color: {CYAN};
                border-color: {CYAN_DIM};
            }}
        """)
        btn_log.clicked.connect(self._open_log_dialog)
        ex_lay.addWidget(btn_log)
        lay.addWidget(ex_card, 1)
        return panel

    def _build_center_panel(self):
        # ── 相机区域始终保持 50/50 比例 ──
        container = QWidget()
        container.setStyleSheet("background: transparent; border: none;")
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        for key, label, color in [
            ("cam1", "● 物品观测区域", CYAN),
            ("cam2", "● 物品放置区",   EMERALD)
        ]:
            card = QWidget()
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            card.setStyleSheet(f"background: {BG_PANEL}; border: 1px solid {BORDER_CLR}; border-radius: 12px;")
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(4, 4, 4, 4)
            card_lay.setSpacing(2)
            lbl_title = QLabel(label)
            lbl_title.setStyleSheet(f"color: {color}; font-size: 19px; background: {BG_DARK};"
                                    f" border-radius: 4px; padding: 2px 8px; border: none;")
            lbl_title.setFixedHeight(35)
            cam_lbl = CamLabel("[ 等待渲染图传数据... ]")
            cam_lbl.setAlignment(Qt.AlignCenter)
            cam_lbl.setStyleSheet(f"color: {TEXT_DIM}; background: {BG_DARK}; border-radius: 8px; border: none;")
            cam_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            card_lay.addWidget(lbl_title)
            card_lay.addWidget(cam_lbl, 1)
            setattr(self, f"cam_label_{key}", cam_lbl)
            lay.addWidget(card, 1)
            
        return container

    def _build_right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setStyleSheet(f"background: {BG_PANEL}; border: 1px solid {BORDER_CLR}; border-radius: 12px;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # 标题栏
        title_bar = QWidget()
        title_bar.setFixedHeight(44)
        title_bar.setStyleSheet(f"background: {BG_CARD}; border-radius: 12px 12px 0 0;")
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(14, 0, 8, 0)
        tb_lay.addWidget(QLabel("💬 对话助手"))
        tb_lay.addStretch()
        btn_clear = QPushButton("🗑️ 清空")
        btn_clear.setFixedHeight(24)
        btn_clear.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: 1px solid {TEXT_DIM};
                color: {TEXT_DIM}; font-size: 17px; border-radius: 5px;
                padding: 0 8px; }}
            QPushButton:hover {{ background: #3f1219; color: {ROSE}; border-color: {ROSE}; }}
            QPushButton:pressed {{ background: #5a1a24; }}
        """)
        btn_clear.clicked.connect(self._clear_chat)
        tb_lay.addWidget(btn_clear)
        lay.addWidget(title_bar)

        # 聊天滚动区
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet(f"background: {BG_DARK}; border: none;")
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet(f"background: {BG_DARK};")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(6)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_container)
        lay.addWidget(self.chat_scroll, 1)

        # 输入区
        input_area = QWidget()
        input_area.setMinimumHeight(72)
        input_area.setStyleSheet(f"background: {BG_CARD}; border-radius: 0 0 12px 12px;")
        in_lay = QHBoxLayout(input_area)
        in_lay.setContentsMargins(10, 10, 10, 10)
        in_lay.setSpacing(6)
        in_lay.setAlignment(Qt.AlignBottom)

        self.input_entry = ChatInputEdit()
        self.input_entry.send_requested.connect(self._on_send)
        in_lay.addWidget(self.input_entry, 1)

        self.btn_sparkle = TactileButton("✨", hover_color=INDIGO)
        self.btn_sparkle.setObjectName("btn_sparkle")
        self.btn_sparkle.setFixedSize(42, 42)
        self.btn_sparkle.setToolTip("润色提示词")
        self.btn_sparkle.clicked.connect(self._handle_smart_plan)
        in_lay.addWidget(self.btn_sparkle)

        self.btn_send = SendButton()
        self.btn_send.clicked.connect(self._on_send)
        in_lay.addWidget(self.btn_send)

        lay.addWidget(input_area)
        return panel

    # ══════════════════════════════════════════════════════
    #  聊天气泡
    # ══════════════════════════════════════════════════════
    def _add_chat_bubble(self, text, side="ai"):
        row = QWidget()
        row.setStyleSheet("background: transparent; border: none;")
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)

        bubble = QLabel(text)

        if side == "system":
            bubble.setWordWrap(False)
            bubble.setStyleSheet(f"""
                background: {BG_CARD}; color: {TEXT_DIM};
                border-radius: 10px; padding: 3px 25px;
                border: 1px solid {BORDER_CLR};
            """)
            row_lay.addStretch()
            row_lay.addWidget(bubble)
            row_lay.addStretch()
        elif side == "user":
            bubble.setObjectName("chat_bubble_user")
            bubble.setWordWrap(True)
            bubble.setMaximumWidth(290)
            bubble.setStyleSheet(f"""
                background: {CYAN_DIM}; color: white;
                border-radius: 14px 14px 2px 14px;
                padding: 8px 14px; border: none;
            """)
            row_lay.addStretch()
            row_lay.addWidget(bubble)
        else:  # ai
            bubble.setObjectName("chat_bubble_ai")
            bubble.setWordWrap(True)
            bubble.setMaximumWidth(290)
            bubble.setStyleSheet(f"""
                background: {BG_CARD}; color: {TEXT_PRI};
                border-radius: 14px 14px 14px 2px;
                padding: 8px 14px;
                border: 1px solid {BORDER_CLR};
            """)
            row_lay.addWidget(bubble)
            row_lay.addStretch()

        # 插入到 stretch 之前
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        # 滚动到底部
        QTimer.singleShot(50, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()))

    def _chat(self, text, side="ai"):
        self.sig.add_chat.emit(text, side)

    def _clear_chat(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        Toast(self.chat_scroll, "🗑️ 对话已清空", success=True)

    # ══════════════════════════════════════════════════════
    #  状态更新
    # ══════════════════════════════════════════════════════
    def _set_status(self, status):
        self.status = status
        if status == "ready":
            self.status_dot.setStyleSheet(f"color: {EMERALD}; font-size: 14px;")
            self.status_label.setStyleSheet(f"color: {EMERALD}; font-size: 13px;")
            self.status_label.setText("系统就绪")
        elif status == "running":
            self.status_dot.setStyleSheet(f"color: {AMBER}; font-size: 14px;")
            self.status_label.setStyleSheet(f"color: {AMBER}; font-size: 13px;")
            self.status_label.setText("任务执行中...")
        else:
            self.status_dot.setStyleSheet(f"color: {ROSE}; font-size: 14px;")
            self.status_label.setStyleSheet(f"color: {ROSE}; font-size: 13px;")
            self.status_label.setText("发生异常")

    def _on_task_done(self):
        self.running = False
        self._set_status("ready")
        self.btn_exec.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._add_chat_bubble("任务完成！", "ai")

    # ══════════════════════════════════════════════════════
    #  相机图像更新
    # ══════════════════════════════════════════════════════
    def _update_camera(self, key, bgr_img):
        lbl = getattr(self, f"cam_label_{key}", None)
        if lbl is None or bgr_img is None:
            return
        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            lbl.set_frame(pix)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════
    #  日志
    # ══════════════════════════════════════════════════════
    def _append_log(self, tag, msg):
        ts = time.strftime("%H:%M:%S")
        self._log_buffer.append((tag, ts, msg))
        # 如果日志窗口打开，实时追加
        if hasattr(self, '_log_text') and self._log_text:
            color_map = {"SUCCESS": EMERALD, "ERR": ROSE, "WARN": AMBER,
                         "STEP": CYAN, "INFO": TEXT_PRI}
            color = color_map.get(tag, TEXT_PRI)
            self._log_text.append(
                f'<span style="color:{TEXT_DIM}">[{ts}]</span> '
                f'<span style="color:{color}">{msg}</span>')
            
            # 根据设置决定是否自动滚动
            from config import Config
            if getattr(Config, 'LOG_AUTOSCROLL', True):
                def _scroll_log():
                    try:
                        if hasattr(self, '_log_text') and self._log_text:
                            self._log_text.verticalScrollBar().setValue(
                                self._log_text.verticalScrollBar().maximum())
                    except RuntimeError:
                        pass
                QTimer.singleShot(10, _scroll_log)

    # ══════════════════════════════════════════════════════
    #  轮询队列（QTimer 驱动）
    # ══════════════════════════════════════════════════════
    def _poll_queues(self):
        # 日志队列
        try:
            while True:
                tag, msg = log_queue.get_nowait()
                if any(k in msg for k in ["[OK]", "完成", "成功"]):
                    tag = "SUCCESS"
                elif msg.startswith("[Step") or msg.startswith("[FUSION"):
                    tag = "STEP"
                elif any(k in msg for k in ["[X]", "ERROR", "错误", "失败"]):
                    tag = "ERR"
                elif any(k in msg for k in ["[!]", "WARNING", "警告"]):
                    tag = "WARN"
                self.sig.add_log.emit(tag, msg)

                if "[Step 1]" in msg:
                    parts = msg.split("抓取目标:")
                    target = parts[1].split("|")[0].strip() if len(parts) > 1 else ""
                    place_parts = msg.split("放置:")
                    place = place_parts[1].strip() if len(place_parts) > 1 else "默认"
                    self.sig.add_chat.emit(
                        f"Step 1：识别抓取目标 → {target}\n放置描述：{place}", "ai")
                elif "[Step 5]" in msg:
                    self.sig.add_chat.emit("Step 5：正在识别放置位置...", "ai")
                elif "放置坐标" in msg:
                    coord = msg.split(":")[-1].strip()
                    self.sig.add_chat.emit(f"放置位置已确定：{coord}", "ai")
                elif tag == "ERR" and "[X]" in msg:
                    self.sig.add_chat.emit(f"执行遇到问题：{msg.replace('[X]','').strip()}", "ai")
        except queue.Empty:
            pass

        # 相机队列（只保留最新帧）
        latest = {}
        try:
            while True:
                item = result_queue.get_nowait()
                latest[item[0]] = item[1]
        except queue.Empty:
            pass
        for key, img in latest.items():
            self.sig.update_cam.emit(key, img)

    # ══════════════════════════════════════════════════════
    #  用户交互
    # ══════════════════════════════════════════════════════
    def _fill_input(self, text):
        self.input_entry.setPlainText(text)
        self.input_entry.setFocus()
        # 移动光标到末尾
        cursor = self.input_entry.textCursor()
        cursor.movePosition(cursor.End)
        self.input_entry.setTextCursor(cursor)

    def _get_mode(self):
        return "smart"

    def _on_send(self):
        text = self.input_entry.toPlainText().strip()
        if not text:
            return
        self.input_entry.clear()
        self._add_chat_bubble(text, "user")

        if self._chat_mode == "clarification":
            self._clarification_result = text
            self._chat_mode = "normal"
            self.sig.set_status.emit("running")
            self._clarification_event.set()
        else:
            self._on_execute_internal(text)

    def _on_execute(self):
        self._on_send()

    def _on_execute_internal(self, instruction):
        if self.running or not self.env_ready:
            if not self.env_ready:
                self._add_chat_bubble("环境尚未就绪，请稍候...", "ai")
            return
        mode = self._get_mode()
        if mode == "smart" and not instruction:
            self._add_chat_bubble("智能放置模式需要输入指令", "ai")
            return
        self.running = True
        self.sig.set_status.emit("running")
        self.btn_exec.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._add_chat_bubble(f"收到指令：{instruction}\n正在分析，请稍候...", "ai")
        self.sig.add_log.emit("STEP",
            f">> 开始执行 | 模式: {mode} | 指令: {instruction or '(无)'}")
        self._task_queue.put((mode, instruction))

    def _on_stop(self):
        self.sig.add_log.emit("WARN", "[!] 用户请求停止（当前步骤完成后生效）")

    def _ask_clarification_blocking(self, question):
        self._clarification_event.clear()
        self._clarification_result = None
        self.sig.add_chat.emit(question, "ai")
        self.sig.set_status.emit("ready")
        self._chat_mode = "clarification"
        self._clarification_event.wait()
        return self._clarification_result or ""

    # ══════════════════════════════════════════════════════
    #  润色提示词
    # ══════════════════════════════════════════════════════
    def _handle_smart_plan(self):
        text = self.input_entry.toPlainText().strip()
        if not text or self.running or self.is_planning:
            return
        self.is_planning = True
        self.btn_sparkle.setEnabled(False)
        self.input_entry.setEnabled(False)
        self.sig.add_log.emit("INFO", f">> 正在润色提示词... 原指令: {text}")

        def _do():
            sys_p = (
                "你是一个机器人指令优化助手。请将用户输入的口语化、模糊或复杂的指令，"
                "润色为专业、明确、更易于视觉语言模型(VLM)理解的机器人抓取指令。\n"
                "要求：1.明确抓取对象。2.明确放置目的地及空间关系。"
                "3.只输出润色后的一句自然语言文本，不要包含任何解释或前缀。"
            )
            self.sig.add_log.emit("INFO", "[润色] 线程启动，准备调用 DeepSeek API...")
            try:
                self.sig.add_log.emit("INFO", "[润色] 正在连接 DeepSeek API ...")
                from config import Config as _Cfg
                result = self._call_llm_api(
                    text, sys_p,
                    api_url=getattr(_Cfg, 'POLISH_BASE_URL', 'https://api.deepseek.com'),
                    api_key=getattr(_Cfg, 'POLISH_API_KEY',  'sk-f668b6a0a68643dea174b74e30ecf9b1'),
                    model=getattr(_Cfg,   'POLISH_MODEL',    'deepseek-chat'),
                )
                self.sig.add_log.emit("INFO", f"[润色] API 返回: {result[:80]}")
                ok = not result.startswith("API 调用失败")
            except Exception as e:
                result = f"API 调用失败: {e}"
                ok = False
                self.sig.add_log.emit("ERR", f"[润色] 异常: {e}")

            self.is_planning = False
            if ok:
                self.sig.add_log.emit("SUCCESS", f">> 润色完成: {result}")
            else:
                self.sig.add_log.emit("ERR", f">> 润色失败: {result}")

            self.sig.plan_done.emit(ok, result)

        threading.Thread(target=_do, daemon=True).start()

    def _on_plan_done(self, ok, result):
        self.input_entry.setEnabled(True)
        self.btn_sparkle.setEnabled(True)
        cw = self.centralWidget()
        if ok:
            self.input_entry.setPlainText(result.strip())
            Toast(cw, "✨ 润色完成", success=True)
        else:
            Toast(cw, f"润色失败：{result[:40]}", success=False)

    def _call_llm_api(self, prompt, system="", api_url=None, api_key=None, model=None):
        try:
            from openai import OpenAI
            import httpx
            _url   = api_url or self._settings.get("api_url", "")
            _key   = api_key or self._settings.get("api_key", "")
            _model = model   or self._settings.get("model_name", "")
            if not _key:
                from config import Config
                _url, _key, _model = (
                    Config.QWEN_BASE_URL, Config.QWEN_API_KEY, Config.QWEN_MODEL)
            print(f"[_call_llm_api] url={_url} model={_model}", flush=True)
            client = OpenAI(api_key=_key, base_url=_url,
                            http_client=httpx.Client(trust_env=False, timeout=30.0))
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            print(f"[_call_llm_api] 发送请求...", flush=True)
            resp = client.chat.completions.create(
                model=_model, messages=msgs, temperature=0.3)
            print(f"[_call_llm_api] 收到响应", flush=True)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[_call_llm_api] 异常: {e}", flush=True)
            return f"API 调用失败: {e}"

    # ══════════════════════════════════════════════════════
    #  设置弹窗
    # ══════════════════════════════════════════════════════
    def _open_settings(self):
        from PyQt5.QtWidgets import QStackedWidget, QSlider, QSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle("设置")
        dlg.resize(580, 420)
        dlg.setStyleSheet(f"background: {BG_DARK}; color: {TEXT_PRI};")
        root = QVBoxLayout(dlg)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        models: dict = {k: dict(v) for k, v in self._settings.get("models", {}).items()}
        if not models:
            models = load_settings()["models"]

        inp_ss = (f"QLineEdit {{ background: {BG_INPUT}; color: {TEXT_PRI};"
                  f" border: 1px solid {BORDER_CLR}; border-radius: 8px;"
                  f" padding: 5px 10px; font-size: 18px; }}"
                  f"QLineEdit:focus {{ border-color: {CYAN_DIM}; }}")
        lbl_ss = f"color: {TEXT_SEC}; font-size: 15px; font-weight: bold; background: transparent; border: none;"

        # ── 左侧导航 + 右侧内容 ──────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(148)
        sidebar.setStyleSheet(f"background: {BG_PANEL}; border-right: 1px solid {BORDER_CLR};")
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(10, 16, 10, 16)
        sb_lay.setSpacing(4)

        stack = QStackedWidget()
        stack.setStyleSheet(f"background: {BG_DARK};")

        nav_active = (f"QPushButton {{ background: {CYAN_DIM}; color: {CYAN};"
                      f" border: none; border-radius: 8px; padding: 10px 12px;"
                      f" text-align: left; font-size: 16px; }}")
        nav_idle   = (f"QPushButton {{ background: transparent; color: {TEXT_SEC};"
                      f" border: none; border-radius: 8px; padding: 10px 12px;"
                      f" text-align: left; font-size: 16px; }}"
                      f"QPushButton:hover {{ background: {BG_CARD}; color: {TEXT_PRI}; }}")

        nav_btns = []
        for label in ["🤖  大模型配置", "🎨  界面设置"]:
            b = QPushButton(label)
            b.setStyleSheet(nav_idle)
            sb_lay.addWidget(b)
            nav_btns.append(b)
        sb_lay.addStretch()

        def switch_page(idx):
            stack.setCurrentIndex(idx)
            for i, b in enumerate(nav_btns):
                b.setStyleSheet(nav_active if i == idx else nav_idle)
            btn_add.setVisible(idx == 0)

        nav_btns[0].clicked.connect(lambda: switch_page(0))
        nav_btns[1].clicked.connect(lambda: switch_page(1))

        body.addWidget(sidebar)
        body.addWidget(stack, 1)
        root.addLayout(body, 1)

        # ── Page 0: 大模型配置 ────────────────────────────
        page_model = QWidget()
        pm = QVBoxLayout(page_model)
        pm.setContentsMargins(22, 20, 20, 16)
        pm.setSpacing(12)

        pm.addWidget(self._ss_label("当前模型", lbl_ss))
        combo_row = QHBoxLayout(); combo_row.setSpacing(6)
        combo = CustomDropdown()
        for name in models:
            combo.addItem(name)
        cur = self._settings.get("current_model", list(models.keys())[0] if models else "")
        idx = combo.findText(cur)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        btn_del = QPushButton("🗑️")
        btn_del.setFixedSize(34, 34)
        btn_del.setToolTip("删除当前模型")
        btn_del.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {TEXT_DIM};
                border: 1px solid {BORDER_CLR}; border-radius: 8px; font-size: 14px; }}
            QPushButton:hover {{ background: #3f1219; color: {ROSE}; border-color: {ROSE}; }}
        """)
        combo_row.addWidget(combo, 1)
        combo_row.addWidget(btn_del)
        pm.addLayout(combo_row)

        pm.addWidget(self._ss_label("模型配置", lbl_ss))

        def make_row(label, widget):
            r = QHBoxLayout(); r.setSpacing(8)
            lbl = QLabel(label)
            lbl.setFixedWidth(72)
            lbl.setStyleSheet(f"color: {TEXT_SEC}; font-size: 17px; background: transparent; border: none;")
            r.addWidget(lbl); r.addWidget(widget, 1)
            return r

        name_e = QLineEdit(); name_e.setStyleSheet(inp_ss); name_e.setFixedHeight(34)
        url_e  = QLineEdit(); url_e.setStyleSheet(inp_ss);  url_e.setFixedHeight(34)
        # key_e  = QLineEdit(); key_e.setEchoMode(QLineEdit.Password) # 遮挡密码
        key_e  = QLineEdit() # 明文显示
        key_e.setStyleSheet(inp_ss); key_e.setFixedHeight(34)
        pm.addLayout(make_row("模型名称", name_e))
        pm.addLayout(make_row("API 地址", url_e))
        pm.addLayout(make_row("API Key",  key_e))
        pm.addStretch()
        stack.addWidget(page_model)

        def fill_fields(model_name):
            m = models.get(model_name, {})
            name_e.setText(model_name)
            url_e.setText(m.get("url", ""))
            key_e.setText(m.get("key", ""))

        fill_fields(combo.currentText())
        combo.currentTextChanged.connect(fill_fields)

        def on_del():
            name = combo.currentText()
            if name in models:
                del models[name]
            combo.removeItem(combo.currentIndex())

        btn_del.clicked.connect(on_del)

        # ── Page 1: 界面设置 ──────────────────────────────
        page_ui = QWidget()
        pu = QVBoxLayout(page_ui)
        pu.setContentsMargins(22, 20, 20, 16)
        pu.setSpacing(25)

        # 字体设置 (丝滑预览优化)
        pu.addWidget(self._ss_label("聊天内容字体缩放", lbl_ss))
        fs_row = QHBoxLayout(); fs_row.setSpacing(10)
        fs_slider = QSlider(Qt.Horizontal)
        fs_slider.setRange(12, 24)
        cur_fs = self._settings.get("font_size", 16)
        fs_slider.setValue(cur_fs)
        fs_slider.setStyleSheet(f"QSlider::groove:horizontal {{ background: {BORDER_CLR}; height: 4px; }} QSlider::handle:horizontal {{ background: {CYAN}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}")
        fs_spin = QSpinBox()
        fs_spin.setRange(12, 24); fs_spin.setValue(cur_fs); fs_spin.setStyleSheet(inp_ss)
        
        # 丝滑预览逻辑：使用单次定时器进行防抖
        self._fs_live_timer = QTimer(dlg)
        self._fs_live_timer.setSingleShot(True)
        self._fs_live_timer.timeout.connect(lambda: QApplication.instance().setStyleSheet(build_stylesheet(fs_slider.value())))

        def on_fs_change(val):
            fs_spin.setValue(val)
            # 20ms 防抖，解决拖动卡顿
            self._fs_live_timer.start(20)
            
        fs_slider.valueChanged.connect(on_fs_change)
        fs_spin.valueChanged.connect(fs_slider.setValue)
        
        fs_row.addWidget(fs_slider, 1); fs_row.addWidget(fs_spin)
        pu.addLayout(fs_row)

        # 图传 FPS 设置 (独立行布局)
        from config import Config as _GlobalCfg
        pu.addWidget(self._ss_label("图传刷新率 (FPS)", lbl_ss))
        fps_line_lay = QHBoxLayout(); fps_line_lay.setSpacing(10)
        fps_slider = QSlider(Qt.Horizontal)
        fps_slider.setRange(1, 30)
        cur_fps = getattr(_GlobalCfg, 'CAMERA_FPS', 15)
        fps_slider.setValue(cur_fps)
        fps_spin_box = QSpinBox()
        fps_spin_box.setRange(1, 30); fps_spin_box.setValue(cur_fps); fps_spin_box.setStyleSheet(inp_ss)
        
        fps_slider.valueChanged.connect(fps_spin_box.setValue)
        fps_spin_box.valueChanged.connect(fps_slider.setValue)
        
        fps_line_lay.addWidget(fps_slider, 1); fps_line_lay.addWidget(fps_spin_box)
        pu.addLayout(fps_line_lay)

        pu.addStretch()
        
        # 恢复默认按钮
        btn_reset = QPushButton("♻️  恢复默认设置")
        btn_reset.setStyleSheet(f"QPushButton {{ background: {BG_INPUT}; color: {TEXT_SEC}; border: 1px solid {BORDER_CLR}; padding: 10px; }} QPushButton:hover {{ color: {AMBER}; border-color: {AMBER}; }}")
        def on_reset():
            fs_slider.setValue(16)
            fps_slider.setValue(15)
            # 实时触发样式回归
            QApplication.instance().setStyleSheet(build_stylesheet(16))
            Toast(dlg, "已重置为默认值", success=True)
        btn_reset.clicked.connect(on_reset)
        pu.addWidget(btn_reset)
        
        stack.addWidget(page_ui)

        # ── 底部按钮栏 ────────────────────────────────────
        btn_bar = QWidget()
        btn_bar.setFixedHeight(60)
        btn_bar.setStyleSheet(f"background: {BG_PANEL}; border-top: 1px solid {BORDER_CLR};")
        bb = QHBoxLayout(btn_bar)
        bb.setContentsMargins(16, 0, 16, 0)
        bb.setSpacing(12)

        btn_add = ActionButton("+ 添加模型", color=EMERALD, dim_color="#064e3b")
        btn_add.setFixedSize(130, 36)
        btn_cancel = QPushButton("退出")
        btn_save = ActionButton("保存配置", color=CYAN, dim_color=CYAN_DIM)
        btn_save.setFixedSize(130, 36)
        
        bb.addWidget(btn_add)
        bb.addStretch()
        bb.addWidget(btn_cancel)
        bb.addWidget(btn_save)
        root.addWidget(btn_bar)

        def on_add():
            _placeholder = "__new__"
            models[_placeholder] = {"url": "", "key": ""}
            combo.addItem(_placeholder)
            combo.setCurrentText(_placeholder)
            name_e.clear()
            url_e.clear(); key_e.clear()
            name_e.setFocus()

        btn_add.clicked.connect(on_add)

        def on_save():
            active_tab = stack.currentIndex()
            cur_name = combo.currentText()
            new_name = name_e.text().strip()
            if not new_name or new_name == "__new__":
                Toast(dlg, "请先填写模型名称", success=False)
                name_e.setFocus()
                return
            entry = {"url": url_e.text().strip(), "key": key_e.text().strip()}
            
            # 更新全局 Config 对象的临时属性
            from config import Config as _GlobalCfg
            _GlobalCfg.CAMERA_FPS = fps_spin_box.value()
            
            if new_name != cur_name and cur_name in models:
                del models[cur_name]
                i = combo.findText(cur_name)
                if i >= 0:
                    combo.setItemText(i, new_name)
            models[new_name] = entry
            sel = new_name
            fs = fs_spin.value()
            try:
                save_to_config(models, sel, fs)
            except Exception as e:
                Toast(dlg, f"保存失败：{str(e)[:40]}", success=False)
                return
            self._settings = load_settings()
            
            # 实时应用配置
            if hasattr(self, '_timer_fps'):
                self._timer_fps = 1.0 / fps_spin_box.value()
                
            # 分页提示
            if active_tab == 0:
                Toast(dlg, "✅ 已保存大模型配置", success=True)
            else:
                Toast(dlg, "✅ 已保存界面设置", success=True)

        btn_save.clicked.connect(on_save)
        btn_cancel.clicked.connect(dlg.reject)
        switch_page(0)
        dlg.exec_()

    @staticmethod
    def _ss_label(text, style):
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    # ══════════════════════════════════════════════════════
    #  调试日志弹窗
    # ══════════════════════════════════════════════════════
    def _open_log_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("系统调试日志 (Debug Console)")
        dlg.resize(800, 560)
        dlg.setStyleSheet(f"background: {BG_DARK}; color: {TEXT_PRI};")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # 标题栏
        hdr = QWidget()
        hdr.setFixedHeight(50)
        hdr.setStyleSheet(f"background: {BG_PANEL}; border-bottom: 1px solid {BORDER_CLR};")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(16, 0, 10, 0)
        hdr_lay.addWidget(QLabel("≡ 系统调试日志"))
        hdr_lay.addStretch()

        btn_copy = TactileButton("📋 复制", hover_color=CYAN)
        btn_copy.setStyleSheet(f"""
            QPushButton {{ background: {INDIGO_DIM}; color: {INDIGO};
                border: 1px solid #4338ca; border-radius: 6px; padding: 4px 12px; font-size: 16px; }}
            QPushButton:hover {{ background: #3730a3; color: white; }}
        """)
        
        btn_clr = TactileButton("🗑️ 清空", hover_color=ROSE)
        btn_clr.setStyleSheet(f"""
            QPushButton {{ background: "#3f1219"; color: {ROSE};
                border: 1px solid {ROSE}; border-radius: 6px; padding: 4px 12px; font-size: 16px; }}
            QPushButton:hover {{ background: "#5a1a24"; color: white; }}
        """)
        
        for b in [btn_copy, btn_clr]:
            hdr_lay.addWidget(b)
        lay.addWidget(hdr)

        # 日志文本
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.setStyleSheet(f"background: {BG_DARK}; color: {TEXT_PRI}; border: none; font-family: monospace; font-size: 17px; padding: 8px;")
        # 回放历史
        for tag, ts, msg in self._log_buffer:
            color_map = {"SUCCESS": EMERALD, "ERR": ROSE, "WARN": AMBER,
                         "STEP": CYAN, "INFO": TEXT_PRI}
            color = color_map.get(tag, TEXT_PRI)
            log_text.append(f'<span style="color:{TEXT_DIM}">[{ts}]</span> <span style="color:{color}">{msg}</span>')
        lay.addWidget(log_text, 1)
        self._log_text = log_text

        def do_copy():
            text = "\n".join(f"[{ts}] [{tag}] {msg}" for tag, ts, msg in self._log_buffer)
            QApplication.clipboard().setText(text)
            btn_copy.setText("✅ 已复制")
            btn_copy.setStyleSheet(f"""
                QPushButton {{ background: {CYAN_DIM}; color: {CYAN};
                    border: 1px solid {CYAN}; border-radius: 6px; padding: 4px 12px; font-size: 16px; }}
            """)
            def _reset_copy():
                try:
                    btn_copy.setText("📋 复制")
                    btn_copy.setStyleSheet(f"""
                        QPushButton {{ background: {INDIGO_DIM}; color: {INDIGO};
                            border: 1px solid #4338ca; border-radius: 6px; padding: 4px 12px; font-size: 16px; }}
                        QPushButton:hover {{ background: #3730a3; color: white; }}
                    """)
                except RuntimeError:
                    pass
            QTimer.singleShot(2000, _reset_copy)
        def do_clear():
            self._log_buffer.clear()
            log_text.clear()
            btn_clr.setText("✅ 已清空")
            btn_clr.setStyleSheet(f"""
                QPushButton {{ background: "#064e3b"; color: {EMERALD};
                    border: 1px solid {EMERALD}; border-radius: 6px; padding: 4px 12px; font-size: 16px; }}
            """)
            def _reset_clr():
                try:
                    btn_clr.setText("🗑️ 清空")
                    btn_clr.setStyleSheet(f"""
                        QPushButton {{ background: "#3f1219"; color: {ROSE};
                            border: 1px solid {ROSE}; border-radius: 6px; padding: 4px 12px; font-size: 16px; }}
                        QPushButton:hover {{ background: "#5a1a24"; color: white; }}
                    """)
                except RuntimeError:
                    pass
            QTimer.singleShot(2000, _reset_clr)

        btn_copy.clicked.connect(do_copy)
        btn_clr.clicked.connect(do_clear)
        dlg.finished.connect(lambda: setattr(self, '_log_text', None))
        dlg.exec_()

    # ══════════════════════════════════════════════════════
    #  MuJoCo 线程（核心逻辑完全保留）
    # ══════════════════════════════════════════════════════
    def _mujoco_loop(self):
        sys.stdout = QueueWriter(log_queue, "INFO")
        sys.stderr = QueueWriter(log_queue, "ERR")
        try:
            from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
            log_queue.put(("INFO", "正在初始化 MuJoCo 环境..."))
            self.env = UR5GraspEnv()
            self.env.reset()
            for _ in range(500):
                self.env.step()
            # 关闭 viewer 窗口：launch_passive 会启动独立渲染线程持续占用 GPU，
            # 与离屏渲染竞争资源导致卡顿，UI 模式下不需要它。
            try:
                self.env.mj_viewer.close()
            except Exception:
                pass
            self.env_ready = True
            log_queue.put(("SUCCESS", "[OK] MuJoCo 环境就绪，启动视频流"))

            # 创建共享任务执行引擎
            from task_executor import TaskExecutor
            self.executor = TaskExecutor(
                env=self.env,
                log_fn=lambda level, msg: log_queue.put((level, msg)),
                ask_fn=self._ask_clarification_blocking,
                image_fn=lambda key, img: result_queue.put((key, img)),
                headless=True,
                render_callback=self._render_for_task,
            )
        except Exception as e:
            import traceback
            log_queue.put(("ERR", f"[X] 环境初始化失败: {e}"))
            log_queue.put(("ERR", traceback.format_exc()))
            return

        while True:
            try:
                mode, instruction = self._task_queue.get_nowait()
                self.running = True
                try:
                    self._do_task(mode, instruction)
                    log_queue.put(("SUCCESS", "[OK] 任务完成"))
                except Exception as e:
                    import traceback
                    log_queue.put(("ERR", f"[X] 任务异常: {e}"))
                    log_queue.put(("ERR", traceback.format_exc()))
                finally:
                    self.running = False
                    self.sig.task_done.emit()
                continue
            except queue.Empty:
                pass
            import mujoco as _mj
            from config import Config
            fps = getattr(Config, 'CAMERA_FPS', 15)
            # 每帧推进足够多步，保持实时仿真速度
            dt = self.env.mj_model.opt.timestep
            steps = max(1, int(round(1.0 / fps / dt)))
            try:
                for _ in range(steps):
                    _mj.mj_step(self.env.mj_model, self.env.mj_data)
                # 不在空闲循环中 sync viewer：viewer.sync() 会等待 viewer 渲染线程
                # 并与离屏渲染争抢 GPU，是卡顿的根本原因。
                # 任务执行时 env.step() 内部会 sync，viewer 仍会在执行时更新。
                for key, cam in [("cam1", "cam"), ("cam2", "cam_global_2")]:
                    imgs  = self.env.render(camera_name=cam)
                    color = cv2.cvtColor(imgs['img'], cv2.COLOR_RGB2BGR)
                    # 不再强制 resize 为 4:3，保持原生 1:1 比例，避免图像“缩水”
                    result_queue.put((key, color))
            except Exception:
                pass
            time.sleep(1.0 / fps)

    def _render_for_task(self):
        try:
            for key, cam in [("cam1", "cam"), ("cam2", "cam_global_2")]:
                imgs = self.env.render(camera_name=cam)
                color = cv2.cvtColor(imgs['img'], cv2.COLOR_RGB2BGR)
                result_queue.put((key, color))
        except Exception:
            pass

    # ══════════════════════════════════════════════════════
    #  任务执行（委托给 TaskExecutor）
    # ══════════════════════════════════════════════════════
    def _do_task(self, mode, instruction):
        self.executor.execute_smart_task(instruction or "")

    def closeEvent(self, event):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        event.accept()


# ══════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    _s = load_settings()
    app.setStyleSheet(build_stylesheet(_s.get("font_size", 13)))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

