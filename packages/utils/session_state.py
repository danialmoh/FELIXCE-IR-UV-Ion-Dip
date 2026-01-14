from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import streamlit as st


def require_state(
    keys: Iterable[str], *, section: str | None = None, hint: str | None = None
) -> bool:
    """Ensure required session_state keys exist before running page logic."""
    missing = [
        key
        for key in keys
        if key not in st.session_state or st.session_state.get(key) is None
    ]
    if missing:
        prefix = f"{section}: " if section else ""
        message = (
            prefix
            + "Please complete the previous step(s) to populate: "
            + ", ".join(missing)
        )
        if hint:
            message += f"\n\n{hint}"
        st.warning(message, icon="⚠️")
        return False
    return True


def get_state_values(keys: Sequence[str]) -> Mapping[str, object]:
    """Convenience helper to grab multiple session_state values at once."""
    return {key: st.session_state.get(key) for key in keys}
