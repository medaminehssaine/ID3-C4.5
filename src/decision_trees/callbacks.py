"""
Observer Pattern for Training Callbacks.

Provides hooks for monitoring and controlling training.
"""
from abc import ABC
from typing import Any, Dict, List, Optional
import time


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of an epoch (for iterative methods)."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        pass

    def on_tree_begin(self, tree_idx: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called before building a tree (for ensembles)."""
        pass

    def on_tree_end(self, tree_idx: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called after building a tree."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self.stop_training = False

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
            if hasattr(cb, 'stop_training') and cb.stop_training:
                self.stop_training = True

    def on_tree_begin(self, tree_idx: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_tree_begin(tree_idx, logs)

    def on_tree_end(self, tree_idx: int, logs: Optional[Dict[str, Any]] = None) -> None:
        for cb in self.callbacks:
            cb.on_tree_end(tree_idx, logs)


class EarlyStopping(Callback):
    """Stop training when a metric stops improving."""

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta

        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True


class ProgressBar(Callback):
    """Display training progress."""

    def __init__(self, total: int):
        self.total = total
        self.current = 0

    def on_tree_end(self, tree_idx: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.current = tree_idx + 1
        pct = (self.current / self.total) * 100
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f'\r[{bar}] {pct:.1f}% ({self.current}/{self.total})', end='', flush=True)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        print()


class MetricsLogger(Callback):
    """Log metrics during training."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.start_time: float = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.history = {}
        self.start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        elapsed = time.time() - self.start_time
        print(f"Training completed in {elapsed:.2f}s")


class ModelCheckpoint(Callback):
    """Save model at intervals or when improved."""

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min'
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.model = None

    def set_model(self, model: Any) -> None:
        self.model = model

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)

        if self.save_best_only and current is not None:
            if self.mode == 'min':
                improved = current < self.best
            else:
                improved = current > self.best

            if improved:
                self.best = current
                self._save_model(epoch)
        elif not self.save_best_only:
            self._save_model(epoch)

    def _save_model(self, epoch: int) -> None:
        if self.model is not None:
            filepath = self.filepath.format(epoch=epoch)
            from .serialization import save_model
            save_model(self.model, filepath)
