import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.gui import MainWindow


def test_main_window_can_be_created():
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert window.windowTitle() == "Movie Clip AI"
        assert window.clips_table.rowCount() == 1
        assert window.start_button.isEnabled()
        assert window.progress_bar.value() == 0
        assert window.auto_select_check.isChecked()
        assert not window.clips_group.isEnabled()
        assert window.clip_count_spin.maximum() == 20
        assert window.quality_combo.currentData() in {
            "fast",
            "balanced",
            "quality",
        }
        assert window.encoder_combo.currentData() in {
            "auto",
            "nvenc",
            "cpu",
        }
        assert window.reframe_combo.currentData() in {
            "smart",
            "center",
            "blurred",
        }
        window._set_busy(True)
        assert not window.start_button.isEnabled()
        assert window.cancel_button.isEnabled()
    finally:
        window.close()
        application.processEvents()
