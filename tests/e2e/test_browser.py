"""Browser-level E2E tests using Playwright.

These tests exercise JavaScript behavior that the Flask test client cannot verify:
localStorage persistence, AJAX round-trips, keyboard shortcuts, debounced saves,
and dynamic DOM manipulation.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


# ── Test 1: Dark mode toggle ──


def test_dark_mode_toggle_persists(e2e_page):
    page, base_url = e2e_page

    # Initially light mode (no data-theme attribute)
    assert page.locator("html").get_attribute("data-theme") is None

    # Click theme toggle
    page.click("#theme-toggle")

    # Verify data-theme="dark" is set on <html>
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")

    # Verify localStorage was set
    theme = page.evaluate("localStorage.getItem('cv-arxiv-theme')")
    assert theme == "dark"

    # Reload page -- theme should persist from localStorage
    page.reload()
    page.wait_for_load_state("networkidle")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")

    # Toggle back to light
    page.click("#theme-toggle")
    assert page.locator("html").get_attribute("data-theme") is None
    theme = page.evaluate("localStorage.getItem('cv-arxiv-theme')")
    assert theme == "light"


# ── Test 2: Paper feedback save/skip button toggle ──


def test_feedback_save_toggles_button(e2e_page):
    page, _base_url = e2e_page

    page.wait_for_selector(".paper-card")

    # Find first paper card's Save button
    save_btn = page.locator(".paper-card").first.locator('.feedback-btn[data-action="save"]')

    # Click save -- should get active styling via AJAX round-trip
    save_btn.click()

    # Wait for the fetch to complete and button to update
    expect(save_btn).to_have_attribute("data-active", "true")

    # Click again -- should toggle off
    save_btn.click()
    expect(save_btn).to_have_attribute("data-active", "")


# ── Test 3: Keyboard shortcuts ──


def test_keyboard_navigation(e2e_page):
    page, _base_url = e2e_page

    page.wait_for_selector(".paper-card")

    # Press 'j' to focus first card (index goes from -1 to 0)
    page.keyboard.press("j")
    first_card = page.locator(".paper-card").first
    expect(first_card).to_have_class(re.compile(r"ring-2"))

    # Press 'j' again to move to second card
    page.keyboard.press("j")
    second_card = page.locator(".paper-card").nth(1)
    expect(second_card).to_have_class(re.compile(r"ring-2"))
    # First card should lose ring
    expect(first_card).not_to_have_class(re.compile(r"ring-2"))

    # Press 'k' to go back to first card
    page.keyboard.press("k")
    expect(first_card).to_have_class(re.compile(r"ring-2"))

    # Press 's' to save the focused card
    page.keyboard.press("s")
    save_btn = first_card.locator('.feedback-btn[data-action="save"]')
    expect(save_btn).to_have_attribute("data-active", "true")

    # Verify '?' shortcut toggles help overlay class
    overlay = page.locator("#shortcut-overlay")
    assert "hidden" in (overlay.get_attribute("class") or "")
    page.evaluate("toggleShortcutHelp()")
    page.wait_for_timeout(100)
    assert "hidden" not in (overlay.get_attribute("class") or "")


# ── Test 4: Debounced notes save ──


def test_notes_debounced_save(e2e_page):
    page, base_url = e2e_page

    page.wait_for_selector(".paper-card")

    # Expand the first card's details
    page.locator(".paper-card").first.locator(".card-toggle").click()

    # Find the notes textarea and type
    notes_textarea = page.locator(".paper-card").first.locator(".notes-textarea")
    expect(notes_textarea).to_be_visible()
    notes_textarea.fill("My research notes here")

    # Wait for debounce (600ms) + network
    page.wait_for_timeout(1500)

    # Reload and verify persistence
    page.goto(f"{base_url}/?timeframe=all")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".paper-card")
    page.locator(".paper-card").first.locator(".card-toggle").click()
    notes_textarea = page.locator(".paper-card").first.locator(".notes-textarea")
    expect(notes_textarea).to_have_value("My research notes here")


# ── Test 5: User tags add/remove ──


def test_add_and_remove_tag(e2e_page):
    page, _base_url = e2e_page

    page.wait_for_selector(".paper-card")

    first_card = page.locator(".paper-card").first
    # User tags live in the expandable details; open them first.
    first_card.locator(".card-toggle").click()
    tags_container = first_card.locator(".user-tags-container")
    expect(tags_container).to_be_visible()

    # Initially no user tags
    initial_tag_count = tags_container.locator(".user-tag").count()

    # Click the add-tag button (the "+ tag" button)
    tags_container.locator("button").click()

    # Type tag name and press Enter
    tag_input = tags_container.locator(".tag-input")
    expect(tag_input).to_be_visible()
    tag_input.fill("important")
    tag_input.press("Enter")

    # Wait for fetch and DOM update
    page.wait_for_timeout(500)

    # Verify tag appeared
    expect(tags_container.locator(".user-tag")).to_have_count(initial_tag_count + 1)
    expect(tags_container.locator(".user-tag").last).to_contain_text("important")

    # Click the tag to remove it
    tags_container.locator(".user-tag").last.click()
    page.wait_for_timeout(500)

    # Verify tag is gone
    expect(tags_container.locator(".user-tag")).to_have_count(initial_tag_count)


# ── Test 6: Bulk selection mode ──


def test_bulk_mode(e2e_page):
    page, _base_url = e2e_page

    page.wait_for_selector(".paper-card")

    # Enter bulk mode
    page.click("#bulk-mode-toggle")

    # Verify checkboxes appear on all paper cards
    cards = page.locator(".paper-card")
    card_count = cards.count()
    assert card_count > 0
    checkboxes = page.locator(".bulk-checkbox")
    expect(checkboxes).to_have_count(card_count)

    # Check two papers
    checkboxes.first.check()
    checkboxes.nth(1).check()

    # Verify bulk action bar appears with "2 selected"
    bulk_bar = page.locator("#bulk-action-bar")
    expect(bulk_bar).to_be_visible()
    expect(bulk_bar).to_contain_text("2 selected")

    # Exit bulk mode
    page.click("#bulk-mode-toggle")
    expect(page.locator(".bulk-checkbox")).to_have_count(0)
    expect(page.locator("#bulk-action-bar")).to_have_count(0)


# ── Test 7: Reading status dropdown ──


def test_reading_status_dropdown(e2e_page):
    page, base_url = e2e_page

    page.wait_for_selector(".paper-card")

    select = page.locator(".paper-card").first.locator(".reading-status-select")
    select.select_option("to_read")

    # Wait for the fetch to complete
    page.wait_for_timeout(500)

    # Reload and verify persistence
    page.goto(f"{base_url}/?timeframe=all")
    page.wait_for_load_state("networkidle")
    page.wait_for_selector(".paper-card")
    select = page.locator(".paper-card").first.locator(".reading-status-select")
    expect(select).to_have_value("to_read")


# ── Test 8: Settings tab navigation ──


def test_settings_tab_navigation(e2e_page):
    page, base_url = e2e_page

    page.goto(f"{base_url}/settings")
    page.wait_for_load_state("networkidle")

    # Default tab should be "interests"
    interests_tab = page.locator('[data-tab="interests"]')
    expect(interests_tab).to_have_attribute("data-active", "true")

    # Click "Ranking" tab (controls)
    controls_tab = page.locator('[data-tab="controls"]')
    controls_tab.click()

    # Verify tab switched
    expect(controls_tab).to_have_attribute("data-active", "true")
    expect(interests_tab).to_have_attribute("data-active", "")


# ── Test 9: Screening a collection from the keyboard ──


def _review_of_every_paper(app) -> int:
    """A collection holding every seeded paper, all unscreened; returns its id."""
    from app.models import Collection, Paper, PaperCollection
    from app.models import db as _db

    with app.app_context():
        collection = Collection(name="E2E Review")
        _db.session.add(collection)
        _db.session.flush()
        for paper in Paper.query.all():
            _db.session.add(PaperCollection(paper_id=paper.id, collection_id=collection.id))
        _db.session.commit()
        return collection.id


def test_screening_keys_in_collection_view(e2e_page, live_server):
    page, base_url = e2e_page
    cid = _review_of_every_paper(live_server["app"])

    page.goto(f"{base_url}/?collection={cid}&timeframe=all&decision=unscreened")
    page.wait_for_load_state("networkidle")
    cards = page.locator(".paper-card")
    expect(cards).to_have_count(3)

    # 'i' on the focused card includes it: it leaves the Unscreened list and focus moves on.
    page.keyboard.press("j")
    page.keyboard.press("i")
    expect(cards).to_have_count(2)
    expect(cards.first).to_have_class(re.compile(r"ring-2"))
    expect(page.locator('[data-decision-count="include"]')).to_have_text("1")
    expect(page.locator('[data-decision-count="unscreened"]')).to_have_text("2")

    page.keyboard.press("e")
    expect(cards).to_have_count(1)
    expect(page.locator('[data-decision-count="all"]')).to_have_text("2")  # the review: excluded is out

    # "All" hides the excluded paper; the included one shows its decision.
    page.goto(f"{base_url}/?collection={cid}&timeframe=all")
    page.wait_for_load_state("networkidle")
    expect(cards).to_have_count(2)
    expect(page.locator('.decision-btn[data-decision="include"][data-active="true"]')).to_have_count(1)
    expect(page.locator("[data-decision-badge]:visible")).to_have_text(["Include"])

    # Clicking the active decision clears it back to unscreened; the card stays in "All".
    page.locator('.decision-btn[data-decision="include"][data-active="true"]').click()
    expect(page.locator('.decision-btn[data-active="true"]')).to_have_count(0)
    expect(cards).to_have_count(2)
    expect(page.locator('[data-decision-count="unscreened"]')).to_have_text("2")


def test_racing_decisions_keep_keyboard_focus_and_keys_only_set(e2e_page, live_server):
    page, base_url = e2e_page
    cid = _review_of_every_paper(live_server["app"])
    page.goto(f"{base_url}/?collection={cid}&timeframe=all&decision=unscreened")
    cards = page.locator(".paper-card")
    expect(cards).to_have_count(3)
    first_id = cards.first.get_attribute("data-paper-id")

    # Two decisions in flight for the focused card (a slow server, 'i' then 'e'): the late
    # response finds it already gone and must not shift focus off the highlighted card.
    page.keyboard.press("j")
    page.keyboard.press("j")
    page.evaluate("""() => {
        const card = getCards()[1];
        const btn = (d) => card.querySelector(`.decision-btn[data-decision="${d}"]`);
        const id = Number(card.dataset.paperId);
        return Promise.all([setDecision(id, "include", btn("include")), setDecision(id, "exclude", btn("exclude"))]);
    }""")
    expect(cards).to_have_count(2)
    page.keyboard.press("i")  # screens the highlighted third paper, not the first
    expect(cards).to_have_count(1)
    expect(cards.first).to_have_attribute("data-paper-id", first_id)

    # A key only sets its decision: pressing it again, or holding another, never re-screens.
    page.goto(f"{base_url}/?collection={cid}&timeframe=all")
    page.keyboard.press("j")
    page.keyboard.press("i")
    include = page.locator('[data-decision-count="include"]')
    expect(include).to_have_text("2")
    page.keyboard.press("i")
    page.evaluate("() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'm', repeat: true }))")
    page.evaluate("() => fetch('/api/collections').then((r) => r.status)")  # serial server: any PUT is done
    page.reload()
    expect(include).to_have_text("2")
    expect(page.locator('[data-decision-count="maybe"]')).to_have_text("0")


def test_next_page_after_screening_does_not_skip_papers(e2e_page, live_server):
    from app.models import db as _db
    from tests.e2e.conftest import _make_paper

    page, base_url = e2e_page
    with live_server["app"].app_context():
        _db.session.add_all(_make_paper(i) for i in range(3, 29))  # 29 unscreened: 24 + 5
        _db.session.commit()
    cid = _review_of_every_paper(live_server["app"])
    page.goto(f"{base_url}/?collection={cid}&timeframe=all&decision=unscreened")
    cards = page.locator(".paper-card")
    expect(cards).to_have_count(24)

    page.keyboard.press("j")
    page.keyboard.press("i")
    expect(cards).to_have_count(23)
    # The screened card's slot went to what was page 2's first paper; Next must not skip it.
    page.get_by_role("link", name="Next", exact=True).click()
    page.wait_for_load_state("networkidle")
    expect(cards).to_have_count(24)


def test_skip_and_remove_keep_screening_chips_live(e2e_page, live_server):
    page, base_url = e2e_page
    cid = _review_of_every_paper(live_server["app"])
    page.goto(f"{base_url}/?collection={cid}&timeframe=all&decision=unscreened")
    cards = page.locator(".paper-card")
    unscreened = page.locator('[data-decision-count="unscreened"]')
    expect(unscreened).to_have_text("3")

    page.keyboard.press("j")
    page.keyboard.press("x")  # skip hides it, and the chips count visible members only
    expect(cards).to_have_count(2)
    expect(unscreened).to_have_text("2")

    page.locator("[data-remove-from-collection]").first.evaluate("(b) => b.click()")
    expect(cards).to_have_count(1)
    expect(unscreened).to_have_text("1")
    expect(page.locator('[data-decision-count="all"]')).to_have_text("1")
