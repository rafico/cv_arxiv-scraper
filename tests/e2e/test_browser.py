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
    expect(save_btn).to_have_class(re.compile(r"bg-emerald-50"))

    # Click again -- should toggle off
    save_btn.click()
    expect(save_btn).not_to_have_class(re.compile(r"bg-emerald-50"))


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
    expect(save_btn).to_have_class(re.compile(r"bg-emerald-50"))

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
    tags_container = first_card.locator(".user-tags-container")

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
    expect(interests_tab).to_have_class(re.compile(r"bg-gray-900"))

    # Click "Ranking" tab (controls)
    controls_tab = page.locator('[data-tab="controls"]')
    controls_tab.click()

    # Verify tab switched
    expect(controls_tab).to_have_class(re.compile(r"bg-gray-900"))
    expect(interests_tab).not_to_have_class(re.compile(r"bg-gray-900"))
