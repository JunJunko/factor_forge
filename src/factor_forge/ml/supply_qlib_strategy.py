"""A-share tradability for the Qlib backtest engine.

Qlib's :class:`Exchange` derives ``limit_buy`` / ``limit_sell`` from ``$change`` and
suspended bars only -- it does NOT model the A-share asymmetry (a limit-UP open blocks
buys, a limit-DOWN open blocks sells; ST / delisting-period / listing-age block buys).
:class:`AShareExchange` overrides :meth:`Exchange._update_limit` to OR in direction-aware
masks built from the immutable panel's point-in-time flags, faithfully mirroring the
project :class:`BacktestEngine` invariants (``engine._can_buy`` / ``_can_sell``).

This module also papers over an environment wart: ``qlib.contrib.strategy.signal_strategy``
unconditionally imports an optimizer that pulls in ``cvxpy``, which is broken against the
installed numpy.  :func:`ensure_topk_strategy_importable` injects a stub optimizer module
so :class:`TopkDropoutStrategy` (which never uses the optimizer) imports cleanly.
"""

from __future__ import annotations

import sys
import types

import pandas as pd

_TOPK_STUB_INSTALLED = False


def ensure_topk_strategy_importable() -> None:
    """Stub out ``qlib.contrib.strategy.optimizer`` so TopkDropoutStrategy imports.

    ``signal_strategy`` imports ``EnhancedIndexingOptimizer`` at module top; that pulls in
    ``cvxpy`` which is incompatible with the installed numpy on this machine.  The
    TopkDropoutStrategy code path never touches the optimizer, so a placeholder class is
    safe.  Idempotent.
    """
    global _TOPK_STUB_INSTALLED
    if _TOPK_STUB_INSTALLED:
        return
    pkg = "qlib.contrib.strategy.optimizer"
    if pkg not in sys.modules:
        try:  # if the real module imports fine, leave it.
            __import__(pkg)
        except Exception:
            stub = types.ModuleType(pkg)
            class _Placeholder:  # never instantiated by TopkDropoutStrategy
                pass
            stub.EnhancedIndexingOptimizer = _Placeholder
            sys.modules[pkg] = stub
    _TOPK_STUB_INSTALLED = True


def build_ashare_masks(
    panel: pd.DataFrame,
    instrument_map: dict[str, str],
    min_listing_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Build ``(limit_buy_mask, limit_sell_mask)`` indexed by ``(instrument, datetime)``.

    A ``True`` value means that side is BLOCKED on that bar, matching the semantics of
    ``Exchange.quote_df['limit_buy']`` / ``['limit_sell']``.
    """
    df = pd.DataFrame({
        "instrument": panel["ts_code"].map(instrument_map).to_numpy(),
        "datetime": pd.to_datetime(panel["trade_date"]).to_numpy(),
        "is_st": panel.get("is_st", False).to_numpy(dtype=bool) if "is_st" in panel else False,
        "is_delisting_period": panel.get("is_delisting_period", False).to_numpy(dtype=bool)
        if "is_delisting_period" in panel else False,
        "is_suspended": panel.get("is_suspended", False).to_numpy(dtype=bool)
        if "is_suspended" in panel else False,
        "is_limit_up_open": panel.get("is_limit_up_open", False).to_numpy(dtype=bool)
        if "is_limit_up_open" in panel else False,
        "is_limit_down_open": panel.get("is_limit_down_open", False).to_numpy(dtype=bool)
        if "is_limit_down_open" in panel else False,
        "listing_trade_days": panel.get("listing_trade_days", 10 ** 9).to_numpy()
        if "listing_trade_days" in panel else 10 ** 9,
    })
    too_young = df["listing_trade_days"] < min_listing_days
    buy_block = df["is_st"] | df["is_delisting_period"] | df["is_limit_up_open"] | df["is_suspended"] | too_young
    sell_block = df["is_limit_down_open"] | df["is_suspended"]
    indexed = df.set_index(["instrument", "datetime"])
    return (
        pd.Series(buy_block.to_numpy(), index=indexed.index, name="limit_buy", dtype=bool),
        pd.Series(sell_block.to_numpy(), index=indexed.index, name="limit_sell", dtype=bool),
    )


def _ensure_exchange_importable():
    from qlib.backtest.exchange import Exchange  # noqa: F401


class AShareExchange:
    """A-share exchange wrapper that ORs panel-derived limit masks into the quote.

    Implemented as a factory returning a dynamic subclass of the qlib ``Exchange`` (fetched
    lazily so importing this module never requires qlib).  The subclass stores the masks
    before ``super().__init__`` runs, then ORs them in inside ``_update_limit``.
    """

    _cls = None

    @classmethod
    def _build_cls(cls):
        if cls._cls is not None:
            return cls._cls
        from qlib.backtest.exchange import Exchange

        class _AShareExchangeImpl(Exchange):
            def __init__(self, *args, limit_buy_mask=None, limit_sell_mask=None, **kwargs):
                # Masks must be set before super().__init__ -- it calls get_quote_from_qlib
                # which calls _update_limit.
                self._ashare_limit_buy = limit_buy_mask
                self._ashare_limit_sell = limit_sell_mask
                super().__init__(*args, **kwargs)

            def _update_limit(self, limit_threshold):  # noqa: D401 - matches qlib signature
                super()._update_limit(limit_threshold)
                q = self.quote_df
                if self._ashare_limit_buy is not None:
                    extra = self._ashare_limit_buy.reindex(q.index).fillna(False).astype(bool).to_numpy()
                    q["limit_buy"] = q["limit_buy"].to_numpy(dtype=bool) | extra
                if self._ashare_limit_sell is not None:
                    extra = self._ashare_limit_sell.reindex(q.index).fillna(False).astype(bool).to_numpy()
                    q["limit_sell"] = q["limit_sell"].to_numpy(dtype=bool) | extra

        _AShareExchangeImpl.__name__ = "AShareExchange"
        cls._cls = _AShareExchangeImpl
        return cls._cls

    @classmethod
    def make(cls, *args, **kwargs):
        """Instantiate the underlying qlib Exchange subclass with A-share limit masks."""
        impl = cls._build_cls()
        return impl(*args, **kwargs)


def load_topk_dropout_strategy():
    """Return ``TopkDropoutStrategy`` (after ensuring the import works despite cvxpy)."""
    ensure_topk_strategy_importable()
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    return TopkDropoutStrategy
