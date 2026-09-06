from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.config import (
    CandidateGenerationSection,
    EncodingSection,
    PathsSection,
    ReframeSection,
    SelectionSection,
    Settings,
    SubtitleSection,
    TranscriptionSection,
    load_settings,
)
from app.exceptions import OperationCancelled
from app.models import ClipRange, RunReport
from app.pipeline import ClipPipeline
from app.services.transcription_service import find_cached_whisper_model
from app.utils.logging_setup import configure_logging
from app.utils.timecodes import parse_clip_range

LOGGER = logging.getLogger(__name__)
VIDEO_FILTER = (
    "Видео (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;"
    "Все файлы (*.*)"
)


@dataclass(frozen=True)
class ProcessingRequest:
    source: Path
    output: Path
    ranges: list[ClipRange]
    settings: Settings
    auto_select: bool
    overwrite: bool
    keep_temp: bool


class PipelineWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    completed = Signal()

    def __init__(self, request: ProcessingRequest) -> None:
        super().__init__()
        self.request = request
        self._cancel_event = threading.Event()

    @Slot()
    def cancel(self) -> None:
        self._cancel_event.set()
        self.log.emit("Запрошена отмена. Текущая операция будет завершена безопасно.")

    def _transcription_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        if total == 100:
            value = 25 + int(20 * current / 100)
        else:
            values = {0: 5, 1: 15, 2: 25, 3: 45}
            value = values.get(current, 25)
        self.progress.emit(min(value, 45), message)
        self.log.emit(message)

    def _render_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        value = 55 + int(40 * current / max(total, 1))
        self.progress.emit(min(value, 95), message)
        self.log.emit(message)

    def _analysis_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        if total == 100:
            value = 46 + int(5 * current / 100)
        else:
            value = 46 + int(9 * current / max(total, 1))
        self.progress.emit(min(value, 55), message)
        self.log.emit(message)

    @Slot()
    def run(self) -> None:
        audio_file: Path | None = None
        try:
            pipeline = ClipPipeline(self.request.settings)
            self.log.emit(f"Исходный файл: {self.request.source}")
            transcription, transcript_file, audio_file = pipeline.transcribe(
                self.request.source,
                overwrite=self.request.overwrite,
                progress_callback=self._transcription_progress,
                cancel_check=self._cancel_event.is_set,
            )
            self.log.emit(f"Транскрипция: {transcript_file.resolve()}")
            selected = []
            ranges = self.request.ranges
            if self.request.auto_select:
                selected = pipeline.analyze(
                    self.request.source,
                    transcription,
                    progress_callback=self._analysis_progress,
                    cancel_check=self._cancel_event.is_set,
                )
                ranges = [candidate.as_range() for candidate in selected]
                if not ranges:
                    raise RuntimeError(
                        "Подходящие фрагменты не найдены. "
                        "Уменьшите минимальную длительность."
                    )
                for candidate in selected:
                    self.log.emit(
                        f"Момент #{candidate.index}: "
                        f"{candidate.start:.1f}-{candidate.end:.1f} сек., "
                        f"оценка {candidate.score:.1f}"
                    )
            report = pipeline.render(
                self.request.source,
                transcription,
                ranges,
                output_root=self.request.output,
                overwrite=self.request.overwrite,
                candidates=selected,
                progress_callback=self._render_progress,
                cancel_check=self._cancel_event.is_set,
            )
            if audio_file.exists() and not self.request.keep_temp:
                audio_file.unlink()
            self.progress.emit(100, "Готово")
            self.succeeded.emit(report)
        except OperationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            LOGGER.exception("GUI processing failed")
            self.failed.emit(str(exc))
        finally:
            self.completed.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings_store = QSettings("MovieClipAI", "MovieClipAI")
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_output: Path | None = None
        self._close_when_done = False
        self._busy = False

        self.setWindowTitle("Movie Clip AI")
        self.setMinimumSize(1040, 700)
        self.resize(1220, 790)
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_ui()
        self._restore_state()
        self._apply_styles()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        open_action = QAction("Выбрать видео", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._choose_video)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 20)
        root.setSpacing(14)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("Movie Clip AI")
        title.setObjectName("title")
        subtitle = QLabel("Вертикальные клипы из локального видео")
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch()
        self.status_label = QLabel("Готов к запуску")
        self.status_label.setObjectName("status")
        header.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignBottom)
        root.addLayout(header)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("divider")
        root.addWidget(divider)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.input_panel = self._create_input_panel()
        execution_panel = self._create_execution_panel()
        splitter.addWidget(self.input_panel)
        splitter.addWidget(execution_panel)
        splitter.setSizes([510, 650])
        root.addWidget(splitter, stretch=1)

        self.setCentralWidget(central)

    def _path_row(
        self,
        line_edit: QLineEdit,
        callback,
        *,
        directory: bool = False,
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(line_edit)
        button = QToolButton()
        icon = (
            QStyle.StandardPixmap.SP_DirOpenIcon
            if directory
            else QStyle.StandardPixmap.SP_DialogOpenButton
        )
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip("Выбрать папку" if directory else "Выбрать видео")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return container

    def _create_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 16, 0)
        layout.setSpacing(12)

        self.source_group = QGroupBox("Файлы")
        source_form = QFormLayout(self.source_group)
        source_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Выберите видео")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Папка результата")
        source_form.addRow(
            "Видео",
            self._path_row(self.source_edit, self._choose_video),
        )
        source_form.addRow(
            "Результат",
            self._path_row(
                self.output_edit,
                self._choose_output,
                directory=True,
            ),
        )
        layout.addWidget(self.source_group)

        self.clips_group = QGroupBox("Ручные фрагменты")
        clips_layout = QVBoxLayout(self.clips_group)
        clips_toolbar = QHBoxLayout()
        clips_toolbar.addStretch()
        add_button = QToolButton()
        add_button.setText("+")
        add_button.setToolTip("Добавить фрагмент")
        add_button.clicked.connect(self._add_clip_row)
        remove_button = QToolButton()
        remove_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        )
        remove_button.setToolTip("Удалить выбранный фрагмент")
        remove_button.clicked.connect(self._remove_clip_rows)
        clips_toolbar.addWidget(add_button)
        clips_toolbar.addWidget(remove_button)
        clips_layout.addLayout(clips_toolbar)

        self.clips_table = QTableWidget(0, 3)
        self.clips_table.setHorizontalHeaderLabels(
            ["Начало", "Конец", "Длительность"]
        )
        self.clips_table.verticalHeader().setVisible(False)
        self.clips_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.clips_table.setAlternatingRowColors(True)
        header = self.clips_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.clips_table.itemChanged.connect(self._refresh_duration)
        self.clips_table.setMinimumHeight(170)
        clips_layout.addWidget(self.clips_table)
        layout.addWidget(self.clips_group)

        self.options_group = QGroupBox("Обработка")
        options = QGridLayout(self.options_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["small", "medium", "large-v3", "turbo"])
        self.language_combo = QComboBox()
        self.language_combo.addItem("Русский", "ru")
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Автоопределение", "")
        self.device_combo = QComboBox()
        self.device_combo.addItem("Автоматически", "auto")
        self.device_combo.addItem("NVIDIA CUDA", "cuda")
        self.device_combo.addItem("Процессор", "cpu")
        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Быстро", "fast")
        self.quality_combo.addItem("Сбалансированно", "balanced")
        self.quality_combo.addItem("Максимальное качество", "quality")
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItem("Автоматически", "auto")
        self.encoder_combo.addItem("NVIDIA NVENC", "nvenc")
        self.encoder_combo.addItem("Процессор", "cpu")
        self.reframe_combo = QComboBox()
        self.reframe_combo.addItem("Следить за лицом", "smart")
        self.reframe_combo.addItem("Обрезать по центру", "center")
        self.reframe_combo.addItem("Размытый фон", "blurred")
        self.auto_select_check = QCheckBox("Автоматически выбирать моменты")
        self.auto_select_check.setChecked(True)
        self.auto_select_check.toggled.connect(self._sync_control_state)
        self.clip_count_spin = QSpinBox()
        self.clip_count_spin.setRange(1, 20)
        self.clip_count_spin.setValue(3)
        self.min_duration_spin = QSpinBox()
        self.min_duration_spin.setRange(10, 180)
        self.min_duration_spin.setValue(20)
        self.min_duration_spin.setSuffix(" сек.")
        self.max_duration_spin = QSpinBox()
        self.max_duration_spin.setRange(10, 180)
        self.max_duration_spin.setValue(60)
        self.max_duration_spin.setSuffix(" сек.")
        self.subtitles_check = QCheckBox("Вшивать субтитры")
        self.subtitles_check.setChecked(True)
        self.overwrite_check = QCheckBox("Пересоздать кеш")
        self.keep_temp_check = QCheckBox("Сохранить WAV")

        options.addWidget(QLabel("Модель Whisper"), 0, 0)
        options.addWidget(self.model_combo, 0, 1)
        options.addWidget(QLabel("Язык"), 1, 0)
        options.addWidget(self.language_combo, 1, 1)
        options.addWidget(QLabel("Устройство"), 2, 0)
        options.addWidget(self.device_combo, 2, 1)
        options.addWidget(QLabel("Качество"), 3, 0)
        options.addWidget(self.quality_combo, 3, 1)
        options.addWidget(QLabel("Кодировщик"), 4, 0)
        options.addWidget(self.encoder_combo, 4, 1)
        options.addWidget(QLabel("Кадрирование"), 5, 0)
        options.addWidget(self.reframe_combo, 5, 1)
        options.addWidget(self.auto_select_check, 6, 0, 1, 2)
        options.addWidget(QLabel("Количество клипов"), 7, 0)
        options.addWidget(self.clip_count_spin, 7, 1)
        options.addWidget(QLabel("Длительность"), 8, 0)
        duration_row = QWidget()
        duration_layout = QHBoxLayout(duration_row)
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.setSpacing(6)
        duration_layout.addWidget(self.min_duration_spin)
        duration_layout.addWidget(QLabel("до"))
        duration_layout.addWidget(self.max_duration_spin)
        options.addWidget(duration_row, 8, 1)
        options.addWidget(self.subtitles_check, 9, 0, 1, 2)
        options.addWidget(self.overwrite_check, 10, 0)
        options.addWidget(self.keep_temp_check, 10, 1)
        options.setColumnStretch(1, 1)
        layout.addWidget(self.options_group)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Запустить")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.start_button.clicked.connect(self._start_processing)
        self.cancel_button = QPushButton("Отменить")
        self.cancel_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        )
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_processing)
        actions.addWidget(self.start_button, stretch=1)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)
        return panel

    def _create_execution_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(12)

        progress_group = QGroupBox("Выполнение")
        progress_layout = QVBoxLayout(progress_group)
        self.stage_label = QLabel("Ожидание")
        self.stage_label.setObjectName("stage")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.stage_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)

        log_group = QGroupBox("Журнал")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(210)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, stretch=1)

        results_group = QGroupBox("Готовые клипы")
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["№", "Начало", "Конец", "Сек.", "Оценка", "Файл"]
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.results_table.doubleClicked.connect(self._open_selected_clip)
        results_header = self.results_table.horizontalHeader()
        for column in range(5):
            results_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        results_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.results_table)
        self.open_output_button = QPushButton("Открыть папку")
        self.open_output_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_output)
        results_layout.addWidget(
            self.open_output_button,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(results_group, stretch=1)
        return panel

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f6f7;
                color: #20272b;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QMenuBar, QMenu {
                background: #ffffff;
            }
            QLabel#title {
                font-size: 25px;
                font-weight: 700;
                color: #172126;
            }
            QLabel#subtitle {
                color: #68767d;
            }
            QLabel#status {
                color: #176b5b;
                font-weight: 600;
            }
            QLabel#stage {
                font-size: 14px;
                font-weight: 600;
            }
            QFrame#divider {
                color: #d7dde0;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7dde0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox, QTextEdit, QTableWidget {
                background: #ffffff;
                border: 1px solid #c9d1d5;
                border-radius: 4px;
                padding: 5px;
                selection-background-color: #176b5b;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QTableWidget:focus {
                border: 1px solid #176b5b;
            }
            QHeaderView::section {
                background: #e8edef;
                border: 0;
                border-right: 1px solid #d2d9dc;
                border-bottom: 1px solid #d2d9dc;
                padding: 6px;
                font-weight: 600;
            }
            QPushButton, QToolButton {
                background: #ffffff;
                border: 1px solid #bcc7cc;
                border-radius: 5px;
                padding: 7px 11px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #edf2f3;
                border-color: #8d9ca3;
            }
            QPushButton:disabled, QToolButton:disabled {
                color: #9aa5aa;
                background: #eef1f2;
            }
            QPushButton#primaryButton {
                background: #176b5b;
                color: #ffffff;
                border-color: #176b5b;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover {
                background: #115b4d;
            }
            QProgressBar {
                background: #e1e7e9;
                border: 0;
                border-radius: 5px;
                height: 18px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #2b8a78;
                border-radius: 5px;
            }
            QSplitter::handle {
                background: #d7dde0;
                width: 1px;
            }
            """
        )

    def _restore_state(self) -> None:
        output = self._settings_store.value(
            "output",
            str(Path("output").resolve()),
        )
        self.output_edit.setText(str(output))
        saved_model = str(self._settings_store.value("model", "medium"))
        model_cache = Path("temp") / "models"
        if not find_cached_whisper_model(saved_model, model_cache):
            saved_model = next(
                (
                    model
                    for model in ("medium", "small", "turbo", "large-v3")
                    if find_cached_whisper_model(model, model_cache)
                ),
                saved_model,
            )
        self.model_combo.setCurrentText(saved_model)
        device = str(self._settings_store.value("device", "auto"))
        device_index = self.device_combo.findData(device)
        self.device_combo.setCurrentIndex(max(0, device_index))
        language = str(self._settings_store.value("language", "ru"))
        language_index = self.language_combo.findData(language)
        self.language_combo.setCurrentIndex(max(0, language_index))
        quality = str(self._settings_store.value("quality", "balanced"))
        quality_index = self.quality_combo.findData(quality)
        self.quality_combo.setCurrentIndex(max(0, quality_index))
        encoder = str(self._settings_store.value("encoder", "auto"))
        encoder_index = self.encoder_combo.findData(encoder)
        self.encoder_combo.setCurrentIndex(max(0, encoder_index))
        reframe = str(self._settings_store.value("reframe", "smart"))
        reframe_index = self.reframe_combo.findData(reframe)
        self.reframe_combo.setCurrentIndex(max(0, reframe_index))
        self._add_clip_row("00:00:00", "00:00:30")
        self.auto_select_check.setChecked(
            self._settings_store.value("auto_select", True, type=bool)
        )
        self.clip_count_spin.setValue(
            int(self._settings_store.value("clip_count", 3))
        )
        self.min_duration_spin.setValue(
            int(self._settings_store.value("min_duration", 20))
        )
        self.max_duration_spin.setValue(
            int(self._settings_store.value("max_duration", 60))
        )
        self._sync_control_state()

    def _save_state(self) -> None:
        self._settings_store.setValue("output", self.output_edit.text())
        self._settings_store.setValue("model", self.model_combo.currentText())
        self._settings_store.setValue(
            "device",
            self.device_combo.currentData(),
        )
        self._settings_store.setValue(
            "language",
            self.language_combo.currentData(),
        )
        self._settings_store.setValue(
            "quality",
            self.quality_combo.currentData(),
        )
        self._settings_store.setValue(
            "encoder",
            self.encoder_combo.currentData(),
        )
        self._settings_store.setValue(
            "reframe",
            self.reframe_combo.currentData(),
        )
        self._settings_store.setValue(
            "auto_select",
            self.auto_select_check.isChecked(),
        )
        self._settings_store.setValue(
            "clip_count",
            self.clip_count_spin.value(),
        )
        self._settings_store.setValue(
            "min_duration",
            self.min_duration_spin.value(),
        )
        self._settings_store.setValue(
            "max_duration",
            self.max_duration_spin.value(),
        )

    @Slot()
    def _choose_video(self) -> None:
        initial = self.source_edit.text() or str(Path("input").resolve())
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            initial,
            VIDEO_FILTER,
        )
        if filename:
            self.source_edit.setText(filename)

    @Slot()
    def _choose_output(self) -> None:
        initial = self.output_edit.text() or str(Path("output").resolve())
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку результата",
            initial,
        )
        if directory:
            self.output_edit.setText(directory)
            self._save_state()

    @Slot()
    def _add_clip_row(
        self,
        start: str = "00:00:00",
        end: str = "00:00:30",
    ) -> None:
        row = self.clips_table.rowCount()
        self.clips_table.insertRow(row)
        self.clips_table.setItem(row, 0, QTableWidgetItem(start))
        self.clips_table.setItem(row, 1, QTableWidgetItem(end))
        duration = QTableWidgetItem("30.0")
        duration.setFlags(duration.flags() & ~Qt.ItemFlag.ItemIsEditable)
        duration.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clips_table.setItem(row, 2, duration)
        self._refresh_duration(self.clips_table.item(row, 1))

    @Slot()
    def _remove_clip_rows(self) -> None:
        rows = sorted(
            {index.row() for index in self.clips_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.clips_table.removeRow(row)
        if self.clips_table.rowCount() == 0:
            self._add_clip_row()

    @Slot(QTableWidgetItem)
    def _refresh_duration(self, item: QTableWidgetItem | None) -> None:
        if item is None or item.column() not in (0, 1):
            return
        row = item.row()
        start_item = self.clips_table.item(row, 0)
        end_item = self.clips_table.item(row, 1)
        duration_item = self.clips_table.item(row, 2)
        if not start_item or not end_item or not duration_item:
            return
        try:
            clip_range = parse_clip_range(
                f"{start_item.text()}..{end_item.text()}"
            )
            duration_item.setText(f"{clip_range.duration:.1f}")
            duration_item.setForeground(Qt.GlobalColor.darkGreen)
        except Exception:
            duration_item.setText("Ошибка")
            duration_item.setForeground(Qt.GlobalColor.darkRed)

    def _collect_ranges(self) -> list[ClipRange]:
        ranges = []
        for row in range(self.clips_table.rowCount()):
            start = self.clips_table.item(row, 0)
            end = self.clips_table.item(row, 1)
            if not start or not end:
                raise ValueError(f"Заполните строку фрагмента {row + 1}.")
            try:
                ranges.append(
                    parse_clip_range(f"{start.text()}..{end.text()}")
                )
            except Exception as exc:
                raise ValueError(
                    f"Неверный фрагмент в строке {row + 1}: {exc}"
                ) from exc
        return ranges

    def _build_request(self) -> ProcessingRequest:
        source = Path(self.source_edit.text().strip())
        if not source.is_file():
            raise ValueError("Выберите существующий видеофайл.")
        output_text = self.output_edit.text().strip()
        if not output_text:
            raise ValueError("Выберите папку результата.")
        output = Path(output_text)
        auto_select = self.auto_select_check.isChecked()
        ranges = [] if auto_select else self._collect_ranges()
        if self.min_duration_spin.value() > self.max_duration_spin.value():
            raise ValueError(
                "Минимальная длительность не может быть больше максимальной."
            )

        settings = load_settings()
        selected_model = self.model_combo.currentText()
        if not find_cached_whisper_model(
            selected_model,
            settings.paths.temp / "models",
        ):
            raise ValueError(
                f"Модель Whisper '{selected_model}' не установлена. "
                "Выберите medium: она уже готова к работе."
            )
        transcription = TranscriptionSection.model_validate(
            settings.transcription.model_dump()
            | {
                "model": selected_model,
                "language": self.language_combo.currentData(),
                "device": self.device_combo.currentData(),
            }
        )
        subtitles = SubtitleSection.model_validate(
            settings.subtitles.model_dump()
            | {"enabled": self.subtitles_check.isChecked()}
        )
        paths = PathsSection.model_validate(
            settings.paths.model_dump() | {"output": output}
        )
        candidate_generation = CandidateGenerationSection.model_validate(
            settings.candidate_generation.model_dump()
            | {
                "min_duration": self.min_duration_spin.value(),
                "max_duration": self.max_duration_spin.value(),
                "target_duration": (
                    self.min_duration_spin.value()
                    + self.max_duration_spin.value()
                )
                // 2,
            }
        )
        selection = SelectionSection.model_validate(
            settings.selection.model_dump()
            | {"clips": self.clip_count_spin.value()}
        )
        encoding = EncodingSection.model_validate(
            settings.encoding.model_dump()
            | {
                "quality_mode": self.quality_combo.currentData(),
                "encoder": self.encoder_combo.currentData(),
            }
        )
        reframe = ReframeSection.model_validate(
            settings.reframe.model_dump()
            | {"mode": self.reframe_combo.currentData()}
        )
        settings = settings.model_copy(
            update={
                "transcription": transcription,
                "subtitles": subtitles,
                "paths": paths,
                "candidate_generation": candidate_generation,
                "selection": selection,
                "encoding": encoding,
                "reframe": reframe,
            }
        )
        return ProcessingRequest(
            source=source.resolve(),
            output=output,
            ranges=ranges,
            settings=settings,
            auto_select=auto_select,
            overwrite=self.overwrite_check.isChecked(),
            keep_temp=self.keep_temp_check.isChecked(),
        )

    @Slot()
    def _start_processing(self) -> None:
        if self._thread and self._thread.isRunning():
            return
        try:
            request = self._build_request()
        except Exception as exc:
            QMessageBox.warning(self, "Проверьте данные", str(exc))
            return

        self._save_state()
        configure_logging(
            request.settings.logging.level,
            request.settings.logging.file,
        )
        self.results_table.setRowCount(0)
        self._last_output = None
        self.open_output_button.setEnabled(False)
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self._append_log("Запуск обработки")

        thread = QThread(self)
        worker = PipelineWorker(request)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.log.connect(self._append_log)
        worker.succeeded.connect(self._on_succeeded)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.completed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot()
    def _cancel_processing(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Отмена...")

    @Slot(int, str)
    def _on_progress(self, value: int, stage: str) -> None:
        self.progress_bar.setValue(value)
        self.stage_label.setText(stage)
        self.status_label.setText(stage)

    @Slot(str)
    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")

    @Slot(object)
    def _on_succeeded(self, report: RunReport) -> None:
        self._last_output = report.output_directory
        self.results_table.setRowCount(len(report.clips))
        for row, clip in enumerate(report.clips):
            values = [
                str(clip.index),
                f"{clip.start:.2f}",
                f"{clip.end:.2f}",
                f"{clip.duration:.2f}",
                (
                    f"{clip.heuristic_score:.1f}"
                    if clip.heuristic_score is not None
                    else "ручной"
                ),
                str(clip.video_file),
            ]
            for column, value in enumerate(values):
                self.results_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(value),
                )
        self.open_output_button.setEnabled(True)
        self.status_label.setText("Готово")
        self.stage_label.setText(
            f"Создано клипов: {len(report.clips)}"
        )
        self._append_log(f"Результат: {report.output_directory}")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Ошибка")
        self.stage_label.setText("Обработка завершилась с ошибкой")
        self._append_log(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка обработки", message)

    @Slot()
    def _on_cancelled(self) -> None:
        self.status_label.setText("Отменено")
        self.stage_label.setText("Обработка отменена")
        self._append_log("Обработка отменена")

    @Slot()
    def _on_thread_finished(self) -> None:
        self._set_busy(False)
        self._worker = None
        self._thread = None
        if self._close_when_done:
            self._close_when_done = False
            self.close()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.source_group.setEnabled(not busy)
        self.options_group.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self._sync_control_state()

    @Slot()
    def _sync_control_state(self) -> None:
        self.clips_group.setEnabled(
            not self._busy and not self.auto_select_check.isChecked()
        )

    @Slot()
    def _open_output(self) -> None:
        if self._last_output and self._last_output.exists():
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self._last_output))
            )

    @Slot()
    def _open_selected_clip(self) -> None:
        row = self.results_table.currentRow()
        if row < 0:
            return
        item = self.results_table.item(row, 5)
        if item:
            QDesktopServices.openUrl(QUrl.fromLocalFile(item.text()))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_file():
                self.source_edit.setText(str(path))
                event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_state()
        if self._thread and self._thread.isRunning():
            answer = QMessageBox.question(
                self,
                "Обработка выполняется",
                "Отменить обработку и закрыть приложение?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._close_when_done = True
                self._cancel_processing()
            event.ignore()
            return
        event.accept()


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("Movie Clip AI")
    application.setOrganizationName("MovieClipAI")
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
