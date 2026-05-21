"""
Optional Weights & Biases (wandb) logging for cellpose training.

This module is intentionally defensive: training must continue normally
whether or not wandb is installed, whether or not the user is logged in,
and whether or not wandb's backend is reachable. Any failure here is
swallowed and downgraded to a log message - it must never raise into the
training loop.

Standard wandb environment variables are honored (e.g. ``WANDB_PROJECT``,
``WANDB_ENTITY``, ``WANDB_API_KEY``, ``WANDB_MODE``, ``WANDB_DIR``,
``WANDB_RUN_GROUP``, ``WANDB_TAGS``). Setting ``WANDB_DISABLED=true`` or
``WANDB_MODE=disabled`` will skip wandb entirely. Setting
``CELLPOSE_DISABLE_WANDB=1`` does the same and is provided for users who
do not want to touch wandb's own env vars.
"""

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT = "cellpose"


def _wandb_disabled_by_env():
    """Honor common opt-out env vars without trying to import wandb."""
    for var in ("CELLPOSE_DISABLE_WANDB", "WANDB_DISABLED"):
        val = os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    if os.environ.get("WANDB_MODE", "").strip().lower() == "disabled":
        return True
    return False


class WandbLogger:
    """
    Thin wrapper around ``wandb`` for cellpose training.

    The logger is considered *enabled* only if:

    1. ``enabled=True`` was passed (the default)
    2. wandb is importable
    3. no opt-out env var is set
    4. ``wandb.init(...)`` completes without raising

    If any of those fail, the logger silently becomes a no-op so that
    ``log()``, ``log_summary()`` and ``finish()`` are always safe to call.

    Project, entity, tags, group, notes, etc. are driven by wandb's own
    environment variables (``WANDB_PROJECT``, ``WANDB_ENTITY``, ``WANDB_TAGS``,
    ``WANDB_RUN_GROUP``, ``WANDB_NOTES``, ...). The project defaults to
    ``"cellpose"`` if ``WANDB_PROJECT`` is not set.
    """

    def __init__(self, enabled=True, run_name=None, config=None):
        self.enabled = False
        self.wandb = None
        self.run = None

        if not enabled:
            return

        if _wandb_disabled_by_env():
            logger.debug("wandb logging skipped (disabled via environment).")
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            logger.debug(
                "wandb not installed; skipping wandb logging. "
                "To enable, `pip install wandb` and `wandb login`."
            )
            return
        except Exception as e:
            logger.debug(f"wandb import failed ({e!r}); skipping wandb logging.")
            return

        # Heuristic check for credentials. wandb.init() will also error out if
        # not logged in, but checking up-front lets us avoid a noisy stack
        # trace in the common "not logged in" case.
        if not self._looks_authenticated(wandb):
            logger.info(
                "wandb is installed but no credentials were found - skipping "
                "wandb logging. Run `wandb login` (or set WANDB_API_KEY) to enable."
            )
            return

        init_kwargs = dict(
            project=os.environ.get("WANDB_PROJECT", _DEFAULT_PROJECT),
            name=run_name,
            config=config or {},
        )
        # The 'reinit' param was migrated to a string value ('finish_previous')
        # in newer wandb. Try the new value first, fall back to the legacy bool.
        try:
            try:
                self.run = wandb.init(reinit="finish_previous", **init_kwargs)
            except TypeError:
                self.run = wandb.init(reinit=True, **init_kwargs)
        except Exception as e:
            logger.warning(
                f"wandb.init() failed ({e!r}); continuing without wandb logging."
            )
            self.run = None
            return

        self.wandb = wandb
        self.enabled = True
        try:
            logger.info(
                f">>> wandb logging enabled: project={self.run.project}, "
                f"run={self.run.name}, url={self.run.url}"
            )
        except Exception:
            logger.info(">>> wandb logging enabled")

    @staticmethod
    def _looks_authenticated(wandb_module):
        """Best-effort check for a usable wandb credential."""
        if os.environ.get("WANDB_API_KEY"):
            return True
        if os.environ.get("WANDB_MODE", "").strip().lower() in ("offline", "dryrun"):
            return True
        try:
            api_key = wandb_module.api.api_key
        except Exception:
            api_key = None
        return bool(api_key)

    def log(self, metrics, step=None, commit=None):
        """Log a dict of scalar metrics. ``None`` values are dropped.
        Safe no-op if disabled."""
        if not self.enabled:
            return
        clean = {k: v for k, v in metrics.items() if v is not None}
        if not clean:
            return
        try:
            kwargs = {}
            if step is not None:
                kwargs["step"] = step
            if commit is not None:
                kwargs["commit"] = commit
            self.wandb.log(clean, **kwargs)
        except Exception as e:
            logger.debug(f"wandb.log() failed ({e!r}); disabling further wandb logs.")
            self.enabled = False

    def log_summary(self, summary):
        """Update the run's summary dict (final/best metrics)."""
        if not self.enabled:
            return
        try:
            for k, v in summary.items():
                if v is None:
                    continue
                self.run.summary[k] = v
        except Exception as e:
            logger.debug(f"wandb summary update failed ({e!r}).")

    def finish(self):
        """End the wandb run if one was created."""
        if not self.enabled:
            return
        try:
            self.wandb.finish()
        except Exception as e:
            logger.debug(f"wandb.finish() failed ({e!r}).")
        finally:
            self.enabled = False
            self.run = None
