import json
import os
import sys

from PyQt5.QtCore import Qt, QRectF, QPointF, QSize, QSizeF, QThreadPool, QRunnable, pyqtSignal, QObject, QTimer, QRect
from PyQt5.QtGui import QIcon, QPixmap, QPainter

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMenuBar,
    QFileDialog,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QScrollArea,
    QGridLayout,
    QSizePolicy,
    QInputDialog,
    QComboBox,
    QDialog,
    QCheckBox,
    QTabWidget,
    QAction,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QSpinBox,
    QGraphicsPixmapItem,
)
from PIL import Image, ImageOps
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")

class DeselectableTextEdit(QTextEdit):
    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.clearFocus()
class CustomMessageBox(QMessageBox):
    def __init__(self, icon, title, text, beep=False, parent=None):
        super(CustomMessageBox, self).__init__(icon, title, text, parent=parent)
        self.beep = beep

    def showEvent(self, event):
        if not self.beep:
            self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
            QTimer.singleShot(0, lambda: self.setWindowFlags(self.windowFlags() & ~Qt.FramelessWindowHint))
        super(CustomMessageBox, self).showEvent(event)
class ImageBrowser(QMainWindow):
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    def show_settings_dialog(self):
        settings_dialog_instance = SettingsDialog(self)
        if settings_dialog_instance.exec_() == QDialog.Accepted:
            pass

    def __init__(self):
        super().__init__()

        self.setWindowTitle("J_Captioneer_v2")
        self.preselected_width = 100
        self.preselected_height = 100
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("J_Captioneer_v2")
        self.settings_dialog = SettingsDialog(self)
        self.directory = None
        self.images = []
        self.current_image = None
        self.text_filename = None
        self.remember_last_directory = True
        self.default_directory = None
        self.thread_pool = QThreadPool()
        self.models = {
            "VIT-GPT2": {
                "name": "VIT-GPT2",
                "model": model,
                "tokenizer": tokenizer,
            },
            "BLIP": {
                "name": "BLIP",
                "model": blip_model,
                "tokenizer": blip_processor,
            },
        }

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QLabel {
                margin: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }

            QLabel:hover {
                border: 1px solid #aaaaaa;
            }

            QLabel:focus {
                border: 1px solid #666666;
            }
        """)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.file_menu = QMenu("File", self.menu_bar)
        self.menu_bar.addMenu(self.file_menu)

        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        self.textbox = DeselectableTextEdit()
        self.textbox.setWordWrapMode(3)
        self.textbox.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum))  # Keep this line

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_text)

        self.left_button = QPushButton("<")
        self.left_button.clicked.connect(self.show_previous_image)

        self.right_button = QPushButton(">")
        self.right_button.clicked.connect(self.show_next_image)

        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.show_thumbnails)

        self.thumbnail_frame = QWidget()
        self.thumbnail_layout = QGridLayout()
        self.thumbnail_layout.setSpacing(10)  # Set spacing between grid items
        self.thumbnail_frame.setLayout(self.thumbnail_layout)
        self.thumbnail_scroll_area = QScrollArea()
        self.thumbnail_scroll_area.setWidget(self.thumbnail_frame)
        self.thumbnail_scroll_area.setWidgetResizable(True)
        self.status_label = QLabel(self)
        self.status_label.setGeometry(QRect(10, 470, 600, 30))
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
        self.status_label.hide()
        self.choose_directory_button = QPushButton("Choose Directory")
        self.choose_directory_button.clicked.connect(self.choose_directory)
        self.layout.addWidget(self.choose_directory_button)

        self.add_prefix_suffix_action = self.file_menu.addAction("Add Prefix/Suffix")
        self.add_prefix_suffix_action.triggered.connect(self.add_prefix_suffix)

        self.find_replace_all_action = self.file_menu.addAction("Find & Replace All")
        self.find_replace_all_action.triggered.connect(self.find_replace_all)

        self.dark_mode_action = self.file_menu.addAction("Dark Mode")
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.toggled.connect(self.toggle_dark_mode)
        self.save_button.setObjectName("saveButton")
        self.generate_captions_button = QPushButton("Generate Captions For All")

        self.crop_resize_button = QPushButton("Crop and Resize")
        self.crop_resize_button.clicked.connect(self.show_crop_resize_dialog)
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItem("VIT-GPT2")
        self.model_dropdown.addItem("BLIP")
        self.generate_captions_button.clicked.connect(self.generate_captions)
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        self.settings_dialog = SettingsDialog()
        self.settings_dialog.load_from_json("settings.json")
        self.load_last_directory()

        
        if self.settings_dialog.dark_mode_on_launch:
            self.toggle_dark_mode(True)

        if self.settings_dialog.remember_last_directory:
            self.load_last_directory()

        self.file_menu.addAction(settings_action)
    def show_status_message(self, message, duration=750):
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #4a9a7d")
        self.status_label.show()
        QTimer.singleShot(duration, self.status_label.hide)


    def predict_step(self, image_paths):
        selected_model = self.model_dropdown.currentText()

        captions = []
        for image_path in image_paths:
            if selected_model == "VIT-GPT2":
                image = feature_extractor(Image.open(image_path).convert("RGB").resize((384, 384)), return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(pixel_values=image["pixel_values"], **self.gen_kwargs)
                caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            elif selected_model == "BLIP":
                raw_image = Image.open(image_path).convert('RGB')
                inputs = blip_processor(raw_image, return_tensors="pt")
                out = blip_model.generate(**inputs)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)

            captions.append(caption)

        return captions

    def load_text(self):
        self.text_filename = os.path.splitext(self.images[self.current_image])[0] + ".txt"

        if os.path.exists(self.text_filename):
            with open(self.text_filename, "r") as file:
                self.textbox.setPlainText(file.read())
        else:
            self.textbox.setPlainText("")

    def show_thumbnails(self):
        self.clear_layout(self.layout)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.choose_directory_button)
        button_layout.addWidget(self.generate_captions_button)
        button_layout.addWidget(self.crop_resize_button)
        self.generate_captions_button.setObjectName("saveButton")

       
        model_layout = QHBoxLayout()
        model_label = QLabel("Captioning Model:")
        model_label.setTextInteractionFlags(Qt.NoTextInteraction)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        button_layout.addLayout(model_layout)

        self.layout.addLayout(button_layout)

        for i in reversed(range(self.thumbnail_layout.count())):
            self.thumbnail_layout.itemAt(i).widget().setParent(None)

        for index, image_path in enumerate(self.images):
            thumbnail_pixmap = QPixmap(image_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.FastTransformation)

            thumbnail_label = QLabel()
            thumbnail_label.setPixmap(thumbnail_pixmap)
            thumbnail_label.setFixedSize(150, 150)
            thumbnail_label.mousePressEvent = lambda event, idx=index: self.show_image(idx)
            self.thumbnail_layout.addWidget(thumbnail_label, index // 4, index % 4)

        self.layout.addWidget(self.thumbnail_scroll_area)

    def create_captioning_settings_tab(self):
        captioning_settings_tab = QWidget()

        model_label = QLabel("Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItem("VIT-GPT2")
        self.model_dropdown.addItem("BLIP")

        layout = QHBoxLayout()
        layout.addWidget(model_label)
        layout.addWidget(self.model_dropdown)
        captioning_settings_tab.setLayout(layout)

        return captioning_settings_tab

    def show_image(self, index):
        self.current_image = index
        image_path = self.images[self.current_image]

        
        pixmap = QPixmap(image_path)
        image_qimage = pixmap.toImage()

        
        max_size = QSize(400, 400)
        scaled_size = image_qimage.size().scaled(max_size, Qt.KeepAspectRatio)

        
        image_qimage_scaled = image_qimage.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_qpixmap = QPixmap.fromImage(image_qimage_scaled)

        self.image_label.setPixmap(image_qpixmap)

        self.clear_layout(self.layout)
        self.layout.addWidget(self.scroll_area)

        control_layout = QVBoxLayout()

        
        generate_caption_button = QPushButton("Generate Caption")
        generate_caption_button.setObjectName("saveButton")
        generate_caption_button.clicked.connect(self.generate_caption_for_current_image)

        
        control_layout.addWidget(generate_caption_button)

        nav_buttons_layout = QHBoxLayout()
        nav_buttons_layout.addWidget(self.left_button)
        nav_buttons_layout.addWidget(self.right_button)

        control_layout.addWidget(self.back_button)
        control_layout.addLayout(nav_buttons_layout)
        control_layout.addWidget(self.textbox)
        control_layout.addWidget(self.save_button)

        self.layout.addLayout(control_layout)

        self.load_text()
        self.image_label.setStyleSheet("border: none;")


    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.setFocus()

    def show_next_image(self):
        if self.current_image is not None and self.current_image < len(self.images) - 1:
            self.show_image(self.current_image + 1)

    def show_previous_image(self):
        if self.current_image is not None and self.current_image > 0:
            self.show_image(self.current_image - 1)

    def choose_directory(self):
        if self.settings_dialog.remember_last_directory and self.directory:
            start_directory = self.directory
        elif self.settings_dialog.default_directory:
            start_directory = self.settings_dialog.default_directory
        else:
            start_directory = ""

        directory = QFileDialog.getExistingDirectory(self, "Choose Directory", start_directory)

        if directory:
            self.directory = directory
            self.load_images()
            self.show_thumbnails()
            if self.settings_dialog.remember_last_directory:
                with open("last_directory.txt", "w") as file:
                    file.write(directory)

    def load_last_directory(self):
        if self.settings_dialog.remember_last_directory:
            try:
                with open("last_directory.txt", "r") as file:
                    last_directory = file.read().strip()
                if os.path.exists(last_directory):
                    self.directory = last_directory
                    self.load_images()
            except FileNotFoundError:
                pass

    def show_crop_resize_dialog(self):
        if not self.images:
            QMessageBox.critical(self, "Error", "No images found in the directory.")
            return

        crop_resize_dialog = CropResizeDialog(self.images,
                                              self.preselected_width,
                                              self.preselected_height, self)
        cropped_successfully = crop_resize_dialog.exec_()
        if cropped_successfully:
            
            self.load_images()

    def load_images(self):
        self.images = []  
        for file in os.listdir(self.directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(self.directory, file)
                self.images.append(image_path)
                
                txt_path = os.path.splitext(image_path)[0] + ".txt"
                if not os.path.exists(txt_path):
                    with open(txt_path, "w") as txt_file:
                        txt_file.write("")

        
        self.show_thumbnails()

    def save_text(self):
        if self.text_filename:
            with open(self.text_filename, "w") as file:
                file.write(self.textbox.toPlainText())
            
            self.show_status_message("Text saved successfully.")
        else:
            
            self.show_status_message("No text file loaded for current image.", duration=5000)



    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            item = layout.takeAt(i)
            if item.widget() is not None:
                item.widget().setParent(None)
            else:
                self.clear_layout(item.layout())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.show_thumbnails()
        elif event.key() == Qt.Key_Left:
            self.show_previous_image()
        elif event.key() == Qt.Key_Right:
            self.show_next_image()
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_text()
        else:
            super().keyPressEvent(event)

    def add_prefix_suffix(self):
        prefix, ok1 = QInputDialog.getText(self, "Prefix", "Enter the prefix:")
        suffix, ok2 = QInputDialog.getText(self, "Suffix", "Enter the suffix:")

        if ok1 and ok2:
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.lower().endswith('.txt'):
                        old_path = os.path.join(root, file)
                        with open(old_path, "r") as f:
                            lines = [prefix + line.strip() + suffix for line in f]
                        with open(old_path, "w") as f:
                            f.write("\n".join(lines))

    def find_replace_all(self):
        find_text, ok1 = QInputDialog.getText(self, "Find", "Enter the text to find:")
        replace_text, ok2 = QInputDialog.getText(self, "Replace", "Enter the text to replace with:")

        if ok1 and ok2:
            for root, _, files in os.walk(self.directory):
                for file in files:
                    if file.lower().endswith('.txt'):
                        old_path = os.path.join(root, file)
                        with open(old_path, "r") as f:
                            content = f.read()
                        content = content.replace(find_text, replace_text)
                        with open(old_path, "w") as f:
                            f.write(content)

    def generate_captions(self):
        if not self.images:
            QMessageBox.critical(self, "Error", "No images found in the directory.")
            return
        selected_model = self.model_dropdown.currentText()
        print(f"Generating captions using the {selected_model} model.")

        
        worker = CaptionWorker(self.images, selected_model, self.predict_step)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.caption_generated.connect(self.save_caption)
        worker.signals.done.connect(self.captions_generated)

        self.thread_pool.start(worker)


    def update_progress(self, progress):
        print(f"Progress: {progress}/{len(self.images)}")

    def save_caption(self, image_path, caption):
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        with open(txt_path, "w") as txt_file:
            txt_file.write(caption)

    def generate_caption_for_current_image(self):
        if self.current_image is not None and self.current_image < len(self.images):
            image_path = self.images[self.current_image]

            
            worker = SingleCaptionWorker(image_path, self.predict_step)
            worker.signals.done.connect(self.single_caption_generated)

            
            self.thread_pool.start(worker)
        else:
            QMessageBox.critical(self, "Error", "No image currently loaded.")

    def captions_generated(self):
        selected_model = self.model_dropdown.currentText()
        QMessageBox.information(self, "Info", f"Captions generated and saved successfully.")
    def single_caption_generated(self):
        QMessageBox.information(self, "Info", "Caption generated and saved for the current image.")
        self.load_text()


    def set_default_directory_from_settings(self):
        if os.path.exists(self.settings_dialog.default_directory):
            self.choose_directory(self.settings_dialog.default_directory)

    def toggle_dark_mode(self, enabled):
            if enabled:
                self.setStyleSheet("""
                    QMainWindow, QWidget, QScrollArea, QTextEdit, QMessageBox {
                        background-color: #262626;
                        color: white;
                    }
                    QPushButton#generateCaptionsButton, QPushButton#settingsSaveButton {
                        background-color: #4a9a7d;
                    }
                    QPushButton#generateCaptionsButton:hover, QPushButton#settingsSaveButton:hover {
                        background-color: #3d816a;
                    }
                    QPushButton#generateCaptionsButton:pressed, QPushButton#settingsSaveButton:pressed {
                        background-color: #3d816a;
                    }
                    QTabWidget::pane, QDialog {
                        background-color: #262626;
                    }
                    QTabBar::tab {
                        background-color: #262626;
                        color: white;
                        padding: 5px;
                    }
                    QTabBar::tab:selected {
                        background-color: #383838;
                    }


                    QLabel {
                        margin: 5px;
                        border: 1px solid #262626;
                        border-radius: 3px;
                    }
                    QLabel:hover {
                        border: 1px solid #aaaaaa;
                    }
                    QLabel:focus {
                        border: 1px solid #666666;
                    }
                    QPushButton {
                        background-color: #505050;
                        border: 1px solid #262626;
                        color: white;
                        border-radius: 3px;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #383838;
                    }
                    QPushButton:pressed {
                        background-color: #383838;
                    }
                    QPushButton#saveButton {
                        background-color: #4a9a7d;
                    }
                    QPushButton#saveButton:hover {
                        background-color: #3d816a;
                    }
                    QPushButton#saveButton:pressed {
                        background-color: #3d816a;
                    }
                    QMenuBar {
                        background-color: #262626;
                        color: white;
                    }
                    QMenuBar::item {
                        background-color: transparent;
                    }
                    QMenuBar::item:selected {
                        background-color: #383838;
                    }
                    QMenu {
                        background-color: #262626;
                        color: white;
                    }
                    QMenu::item:selected {
                        background-color: #383838;
                    }
                """)
            else:
                self.setStyleSheet("""
                    QLabel {
                        margin: 5px;
                        border: 1px solid #cccccc;
                        border-radius: 3px;
                    }
                    QLabel:hover {
                        border: 1px solid #aaaaaa;
                    }
                    QLabel:focus {
                        border: 1px solid #262626;
                    }
                """)
class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    caption_generated = pyqtSignal(str, str)
    done = pyqtSignal()


class CaptionWorker(QRunnable):
    def __init__(self, images, selected_model, predict_step):
        super(CaptionWorker, self).__init__()
        self.images = images
        self.selected_model = selected_model
        self.predict_step = predict_step
        self.signals = WorkerSignals()

    def run(self):
        for index, image_path in enumerate(self.images):
            caption = self.predict_step([image_path])[0]
            self.signals.caption_generated.emit(image_path, caption)
            self.signals.progress.emit(index + 1)
        self.signals.done.emit()

class SingleCaptionWorker(QRunnable):
    def __init__(self, image_path, predict_step):
        super(SingleCaptionWorker, self).__init__()
        self.image_path = image_path
        self.predict_step = predict_step
        self.signals = WorkerSignals()

    def run(self):
        caption = self.predict_step([self.image_path])[0]
        txt_path = os.path.splitext(self.image_path)[0] + ".txt"
        with open(txt_path, "w") as txt_file:
            txt_file.write(caption)
        self.signals.done.emit()


class CropResizeDialog(QDialog):
    def __init__(self, images, preselected_width=100, preselected_height=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop & Resize Images")
        self.images = images
        self.preselected_width = preselected_width
        self.preselected_height = preselected_height
        self.layout = QVBoxLayout(self)
        self.init_ui()

    def init_ui(self):
        
        self.width_label = QLabel("Width:")
        self.height_label = QLabel("Height:")
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 10000)
        self.width_input.setValue(self.preselected_width)
        self.height_input = QSpinBox()
        self.height_input.setRange(1, 10000)
        self.height_input.setValue(self.preselected_height)

        
        self.width_input.valueChanged.connect(self._updateAspectRatio)
        self.height_input.valueChanged.connect(self._updateAspectRatio)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.width_label)
        input_layout.addWidget(self.width_input)
        input_layout.addWidget(self.height_label)
        input_layout.addWidget(self.height_input)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.preview_container = QWidget()
        self.preview_layout = QGridLayout(self.preview_container)
        self.preview_container.setLayout(self.preview_layout)  # Set the layout for the preview_container widget
        self.preview_layout.setHorizontalSpacing(10)
        self.preview_layout.setVerticalSpacing(10)

        self.scroll_area.setWidget(self.preview_container)

        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("saveButton")
        self.cancel_button = QPushButton("Cancel")
        self.save_button.clicked.connect(self.crop_and_resize_and_close)
        self.cancel_button.clicked.connect(self.reject)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        self.layout.addLayout(input_layout)
        self.layout.addWidget(self.scroll_area)
        self.layout.addLayout(button_layout)

        self.load_images()

        self.adjustSize()

    def load_images(self):
        self.scenes = []
        self.selection_boxes = []
        row, col = 0, 0

        max_width, max_height = 200, 200

        for image_path in self.images:
            
            scene = QGraphicsScene()
            self.scenes.append(scene)

            pixmap = QPixmap(image_path)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.addItem(pixmap_item)

            
            scene.setSceneRect(pixmap_item.boundingRect())

            
            selection_box = DraggableRectItem(0, 0,
                                              pixmap.size().width(),
                                              pixmap.size().height())
            selection_box.setFlag(QGraphicsItem.ItemIsSelectable, True)
            selection_box.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.selection_boxes.append(selection_box)

            scene.addItem(selection_box)

            
            selection_box.set_aspect_ratio(
                    self.preselected_width / float(self.preselected_height))

            
            graphics_view = QGraphicsView()
            graphics_view.setRenderHint(QPainter.Antialiasing)
            graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
            graphics_view.setRenderHint(QPainter.TextAntialiasing)
            graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
            graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            graphics_view.setScene(scene)

            
            original_size = pixmap.size()
            scaled_size = original_size.scaled(max_width, max_height, Qt.KeepAspectRatio)
            scale_factor = scaled_size.width() / original_size.width()
            graphics_view.setFixedSize(scaled_size)

            
            graphics_view.scale(scale_factor, scale_factor)

            
            self.preview_layout.addWidget(graphics_view, row, col)
            col += 1
            if col % 4 == 0:
                row += 1
                col = 0

    def _updateAspectRatio(self, value):
        """
        Loop over the selection boxes and update their aspect ratios.
        """
        aspect_ratio = self.width_input.value() / float(self.height_input.value())
        for selection_box in self.selection_boxes:
            selection_box.set_aspect_ratio(aspect_ratio)

    def crop_and_resize_and_close(self):
        self.crop_and_resize_images()
        self.accept()

    def crop_and_resize_images(self):
        try:
            self.cropped_images = [self._process_image(image_path,
                                                       self.selection_boxes[i],
                                                       self.width_input.value(),
                                                       self.height_input.value())
                                    for i, image_path in enumerate(self.images)]
        except Exception as e:
            print(f"Error processing images: {e}")

    def _process_image(self, image_path, selection_box, output_width, output_height):
        with Image.open(image_path) as image:
            
            crop_x = selection_box.pos().x()
            crop_y = selection_box.pos().y()
            crop_width = selection_box.rect().width()
            crop_height = selection_box.rect().height()

            
            crop_box = (crop_x, crop_y, crop_x + crop_width, crop_y + crop_height)
            cropped_image = image.crop(crop_box)

            
            resized_image = cropped_image.resize((output_width, output_height),
                                                 Image.ANTIALIAS)

            
            resized_image.save(image_path, image.format)


class DraggableRectItem(QGraphicsRectItem):
    def __init__(self, x, y, max_width, max_height):
        super(DraggableRectItem, self).__init__(x, y, max_width, max_height)
        self.max_width = max_width
        self.max_height = max_height
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event):
        
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)

        super(DraggableRectItem, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.OpenHandCursor)

        super(DraggableRectItem, self).mouseReleaseEvent(event)

    def set_aspect_ratio(self, aspect_ratio):
        """
        Update the dimensions of the rect item to match the aspect ratio
        provided.
        """
        
        new_width = self.rect().height() * aspect_ratio

        
        new_size = QSizeF(new_width, self.rect().height())
        new_size = new_size.scaled(self.max_width, self.max_height, Qt.KeepAspectRatio)

        
        current_rect = self.rect()
        self.setRect(QRectF(current_rect.x(), current_rect.y(),
                            new_size.width(), new_size.height()))

        
        self.setPos(self._getSafePosition(self.pos()))

    def set_rect_width(self, width):
        current_rect = self.rect()
        self.setRect(QRectF(current_rect.x(), current_rect.y(), width, current_rect.height()))

        
        self.setPos(self._getSafePosition(self.pos()))

    def set_rect_height(self, height):
        current_rect = self.rect()
        self.setRect(QRectF(current_rect.x(), current_rect.y(), current_rect.width(), height))

        
        self.setPos(self._getSafePosition(self.pos()))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            return self._getSafePosition(value)

        return super(DraggableRectItem, self).itemChange(change, value)

    def _getSafePosition(self, position):
        """
        Given a position for the draggable rectangle, return a position that
        is within the bounds of the scene the rectangle exists within.
        """
        new_position = position
        box_rect = self.rect()
        scene_rect = self.scene().sceneRect()

        
        if position.x() < 0:
            new_position.setX(0)
        elif position.x() + box_rect.width() > scene_rect.width():
            new_position.setX(scene_rect.width() - box_rect.width())

        if position.y() < 0:
            new_position.setY(0)
        elif position.y() + box_rect.height() > scene_rect.height():
            new_position.setY(scene_rect.height() - box_rect.height())

        return new_position


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        ...
        self.setWindowTitle("Settings")
        self.load_from_json("settings.json")
        self.dark_mode_on_launch = False
        self.remember_last_directory = False

        self.tab_widget = QTabWidget()


        self.tab_widget.addTab(self.create_general_settings_tab(), "General")


        save_button = QPushButton("Save")
        save_button.setObjectName("saveButton")
        save_button.clicked.connect(self.save_settings)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)


        button_layout = QHBoxLayout()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)


        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
    def load_from_json(self, filename):
        try:
            with open(filename, "r") as file:
                settings = json.load(file)
            self.dark_mode_on_launch = settings.get("dark_mode_on_launch", False)
            self.default_directory = settings.get("default_directory", "")
            self.remember_last_directory = settings.get("remember_last_directory", False)
        except FileNotFoundError:

            self.dark_mode_on_launch = False
            self.default_directory = ""
            self.remember_last_directory = False
    def save_to_json(self, filename):
        settings = {
            "dark_mode_on_launch": self.dark_mode_on_launch,
            "default_directory": self.default_directory,
            "remember_last_directory": self.remember_last_directory
        }
        with open(filename, "w") as file:
            json.dump(settings, file, indent=2)
    def save_settings(self):
        self.dark_mode_on_launch = self.dark_mode_checkbox.isChecked()
        self.remember_last_directory = self.remember_last_directory_checkbox.isChecked()


        self.save_to_json("settings.json")

        self.accept()
    def change_model(self):
        selected_model = self.model_dropdown.currentText()
        self.parent().model = self.parent().models[selected_model]["model"]
        self.parent().tokenizer = self.parent().models[selected_model]["tokenizer"]
        print(f"Model changed to {selected_model}.")

    def create_general_settings_tab(self):
        general_settings_tab = QWidget()

        self.dark_mode_checkbox = QCheckBox("Enable Dark Mode on Launch")
        self.dark_mode_checkbox.setChecked(self.dark_mode_on_launch)

        self.default_directory_button = QPushButton("Set Default Directory")
        self.default_directory_button.clicked.connect(self.set_default_directory)

        self.remember_last_directory_checkbox = QCheckBox("Remember Last Directory")
        self.remember_last_directory_checkbox.setChecked(self.remember_last_directory)
        self.remember_last_directory_checkbox.toggled.connect(self.default_directory_button.setDisabled)

        layout = QVBoxLayout()
        layout.addWidget(self.dark_mode_checkbox)
        layout.addWidget(self.default_directory_button)
        layout.addWidget(self.remember_last_directory_checkbox)
        general_settings_tab.setLayout(layout)

        return general_settings_tab

    def show_settings_dialog(self):
        settings_dialog_instance = SettingsDialog(self)
        settings_dialog_instance.exec_()

    def set_default_directory(self):
        default_directory = QFileDialog.getExistingDirectory(self, "Select Default Directory")
        if default_directory:
            self.default_directory = default_directory

if __name__ == "__main__":
    app = QApplication(sys.argv)
    image_browser = ImageBrowser()
    image_browser.setGeometry(100, 100, 800, 600)
    image_browser.show()
    sys.exit(app.exec_())
