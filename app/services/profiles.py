"""Multiple named interest profiles (Wave 3).

Each :class:`app.models.InterestProfile` row owns its own learned ranker
artifact, feed ordering, and digest section, plus an editable natural-language
description blended into scoring. This module is the single source of truth for
the profile invariants:

* exactly one profile is ``is_default`` (created by the schema bootstrap and
  never deletable),
* exactly one profile is ``is_active`` (the one the dashboard/scoring uses),
* the default or the last remaining profile can never be deleted.

Pre-Wave-3 feedback rows carry ``profile_id IS NULL``; the default profile owns
them by definition, so training/centroid queries for the default profile union
``profile_id = default.id`` with ``profile_id IS NULL`` (see
``app.services.learned_ranker`` / ``app.services.interest_model``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import InterestProfile, db

DEFAULT_PROFILE_NAME = "Default"
DEFAULT_PROFILE_SLUG = "default"
_MAX_NAME_LEN = 128
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ProfileRef:
    """Lightweight, app-context-free handle to the active profile.

    Snapshotted into the learned-ranker runtime state so scrape worker threads
    (no app context) can resolve the active profile's artifact filename and NL
    description without a DB hit.
    """

    id: int | None
    slug: str
    is_default: bool
    description: str

    @classmethod
    def default(cls) -> ProfileRef:
        return cls(id=None, slug=DEFAULT_PROFILE_SLUG, is_default=True, description="")


def slugify(name: str) -> str:
    """Slugify a profile name to a URL/file-safe token (ASCII lowercase + dashes)."""
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or "profile"


def _unique_slug(name: str, *, exclude_id: int | None = None) -> str:
    base = slugify(name)
    candidate = base
    suffix = 2
    while True:
        query = InterestProfile.query.filter_by(slug=candidate)
        if exclude_id is not None:
            query = query.filter(InterestProfile.id != exclude_id)
        if query.first() is None:
            return candidate
        candidate = f"{base}-{suffix}"
        suffix += 1


def ensure_default_profile() -> InterestProfile:
    """Return the default profile, creating it (and one active profile) if needed.

    Idempotent. Repairs the invariants opportunistically: if no profile is
    default it promotes the earliest one; if none is active it activates the
    default.
    """
    profiles = InterestProfile.query.order_by(InterestProfile.id.asc()).all()
    if not profiles:
        profile = InterestProfile(
            name=DEFAULT_PROFILE_NAME,
            slug=DEFAULT_PROFILE_SLUG,
            description="",
            is_default=True,
            is_active=True,
            include_in_digest=True,
        )
        db.session.add(profile)
        db.session.commit()
        return profile

    default = next((p for p in profiles if p.is_default), None)
    if default is None:
        default = profiles[0]
        default.is_default = True
    if not any(p.is_active for p in profiles):
        default.is_active = True
    db.session.commit()
    return default


def list_profiles() -> list[InterestProfile]:
    ensure_default_profile()
    return InterestProfile.query.order_by(InterestProfile.is_default.desc(), InterestProfile.created_at.asc()).all()


def get_profile(profile_id: int) -> InterestProfile | None:
    return db.session.get(InterestProfile, profile_id)


def get_default_profile() -> InterestProfile:
    return ensure_default_profile()


def get_active_profile() -> InterestProfile:
    """Return the single active profile (repairing/creating as needed)."""
    default = ensure_default_profile()
    active = InterestProfile.query.filter_by(is_active=True).order_by(InterestProfile.id.asc()).first()
    if active is None:
        default.is_active = True
        db.session.commit()
        return default
    return active


def active_profile_ref() -> ProfileRef:
    """DB-backed snapshot of the active profile for the runtime cache."""
    profile = get_active_profile()
    return ProfileRef(
        id=int(profile.id),
        slug=str(profile.slug),
        is_default=bool(profile.is_default),
        description=str(profile.description or ""),
    )


def profile_ref(profile: InterestProfile) -> ProfileRef:
    return ProfileRef(
        id=int(profile.id),
        slug=str(profile.slug),
        is_default=bool(profile.is_default),
        description=str(profile.description or ""),
    )


def create_profile(name: str, description: str = "") -> InterestProfile:
    """Create a new (inactive) profile. Raises ``ValueError`` on a blank name."""
    ensure_default_profile()
    clean = (name or "").strip()[:_MAX_NAME_LEN]
    if not clean:
        raise ValueError("Profile name is required")
    profile = InterestProfile(
        name=clean,
        slug=_unique_slug(clean),
        description=(description or "").strip(),
        is_default=False,
        is_active=False,
        include_in_digest=True,
    )
    db.session.add(profile)
    db.session.commit()
    return profile


def rename_profile(profile_id: int, name: str) -> InterestProfile:
    profile = get_profile(profile_id)
    if profile is None:
        raise LookupError(f"Profile {profile_id} not found")
    clean = (name or "").strip()[:_MAX_NAME_LEN]
    if not clean:
        raise ValueError("Profile name is required")
    profile.name = clean
    profile.slug = _unique_slug(clean, exclude_id=profile.id)
    db.session.commit()
    return profile


def update_description(profile_id: int, description: str) -> InterestProfile:
    profile = get_profile(profile_id)
    if profile is None:
        raise LookupError(f"Profile {profile_id} not found")
    profile.description = (description or "").strip()
    db.session.commit()
    return profile


def set_active_profile(profile_id: int) -> InterestProfile:
    """Make ``profile_id`` the sole active profile."""
    profile = get_profile(profile_id)
    if profile is None:
        raise LookupError(f"Profile {profile_id} not found")
    for other in InterestProfile.query.filter(InterestProfile.is_active.is_(True)).all():
        if other.id != profile.id:
            other.is_active = False
    profile.is_active = True
    db.session.commit()
    return profile


def set_include_in_digest(profile_id: int, value: bool) -> InterestProfile:
    profile = get_profile(profile_id)
    if profile is None:
        raise LookupError(f"Profile {profile_id} not found")
    profile.include_in_digest = bool(value)
    db.session.commit()
    return profile


def delete_profile(profile_id: int) -> None:
    """Delete a profile, forbidding removal of the default or the last one.

    Feedback rows tagged to the deleted profile are re-homed to the default
    profile (never destroyed — data-loss history). If the active profile is
    deleted, activation falls back to the default.
    """
    profile = get_profile(profile_id)
    if profile is None:
        raise LookupError(f"Profile {profile_id} not found")
    if profile.is_default:
        raise ValueError("The default profile cannot be deleted")
    if InterestProfile.query.count() <= 1:
        raise ValueError("Cannot delete the last remaining profile")

    from app.models import PaperFeedback

    default = ensure_default_profile()
    PaperFeedback.query.filter_by(profile_id=profile.id).update(
        {PaperFeedback.profile_id: default.id}, synchronize_session=False
    )
    was_active = bool(profile.is_active)
    db.session.delete(profile)
    if was_active:
        default.is_active = True
    db.session.commit()


def profile_to_dict(profile: InterestProfile) -> dict:
    return {
        "id": int(profile.id),
        "name": profile.name,
        "slug": profile.slug,
        "description": profile.description or "",
        "is_default": bool(profile.is_default),
        "is_active": bool(profile.is_active),
        "include_in_digest": bool(profile.include_in_digest),
    }
