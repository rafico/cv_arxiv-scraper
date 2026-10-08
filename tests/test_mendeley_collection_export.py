"""Sending a whole collection to a Mendeley folder (Mendeley's API is faked in memory)."""

from __future__ import annotations

import threading
from datetime import date, datetime, timezone
from unittest.mock import patch

import requests
from sqlalchemy import inspect, text

from app.models import Collection, Paper, PaperCollection, db
from app.services.mendeley import EXPORT_BATCH_SIZE
from tests.helpers import FakeMendeley, FlaskDBTestCase

CONNECTED = {"status": "connected", "message": "Mendeley is connected."}
MEMBERS = EXPORT_BATCH_SIZE + 2  # visible members in review: one full batch and a short one


def _paper(idx: int, **overrides) -> Paper:
    today = date.today()
    defaults = dict(
        arxiv_id=f"2610.{5000 + idx:04d}",
        title=f"Export Paper {idx}",
        authors="Ada Lovelace, Alan Turing",
        link=f"https://arxiv.org/abs/2610.{5000 + idx:04d}",
        pdf_link=f"https://arxiv.org/pdf/2610.{5000 + idx:04d}",
        abstract_text="abstract",
        match_type="Title",
        is_hidden=False,
        publication_date=today.isoformat(),
        publication_dt=today,
        scraped_date=today.isoformat(),
        scraped_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    defaults.update(overrides)
    return Paper(**defaults)


class MendeleyCollectionExportTests(FlaskDBTestCase):
    def setUp(self):
        super().setUp()
        self.client = self.app.test_client()
        self.mendeley = FakeMendeley()
        for target, replacement in (
            ("app.services.mendeley.requests.get", self.mendeley.get),
            ("app.services.mendeley.requests.post", self.mendeley.post),
            ("app.services.mendeley.MendeleyClient._load_token", lambda _client: {"access_token": "token"}),
        ):
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("app.services.mendeley.MendeleyClient.check_connection", return_value=CONNECTED)
        self.check_connection = patcher.start()
        self.addCleanup(patcher.stop)

        # MEMBERS papers in review, then an excluded member, a hidden member and an outsider.
        self.papers = [_paper(i) for i in range(MEMBERS + 3)]
        self.papers[MEMBERS + 1].is_hidden = True
        self.collection = Collection(name="Thermal detection")
        db.session.add_all([*self.papers, self.collection])
        db.session.flush()
        for paper in self.papers[: MEMBERS + 2]:
            decision = "exclude" if paper is self.papers[MEMBERS] else None
            db.session.add(PaperCollection(paper_id=paper.id, collection_id=self.collection.id, decision=decision))
        db.session.commit()
        self.cid = self.collection.id
        self.members = self.papers[:MEMBERS]

        self.client.get("/")
        with self.client.session_transaction() as session:
            self.csrf = {"X-CSRF-Token": session["settings_csrf_token"]}
        self.check_connection.reset_mock()  # the dashboard probed it once to decide on its buttons

    def _post(self, after: object = 0):
        return self.client.post(f"/api/collections/{self.cid}/mendeley", json={"after": after}, headers=self.csrf)

    def _export(self) -> list[dict]:
        """One pass the way the collection page runs it; every batch's reply."""
        replies: list[dict] = []
        after: int | None = 0
        while after is not None:
            response = self._post(after)
            self.assertEqual(response.status_code, 200, response.get_json())
            replies.append(response.get_json())
            after = replies[-1]["after"]
        return replies

    def _folder_arxiv_ids(self, folder_id: str) -> list[str]:
        return sorted(
            self.mendeley.documents[doc_id]["identifiers"]["arxiv"]
            for doc_id in self.mendeley.folder_documents[folder_id]
        )

    def _posts(self) -> list[tuple[str, str]]:
        return [call for call in self.mendeley.calls if call[0] == "post"]

    # ── the export ──

    def test_first_export_fills_a_new_folder_batch_by_batch(self):
        replies = self._export()

        self.assertEqual([reply["sent"] for reply in replies], [EXPORT_BATCH_SIZE, 2])
        self.assertEqual([reply["remaining"] for reply in replies], [2, 0])
        self.assertEqual({reply["total"] for reply in replies}, {MEMBERS})
        self.assertEqual([reply["failures"] for reply in replies], [[], []])
        self.assertEqual(replies[0]["after"], self.members[EXPORT_BATCH_SIZE - 1].id)

        # One folder named like the collection, holding the review's visible papers only.
        self.assertEqual(self.mendeley.folders, {"folder-1": "Thermal detection"})
        self.assertEqual(self._folder_arxiv_ids("folder-1"), [paper.arxiv_id for paper in self.members])
        self.assertEqual(len(self.mendeley.documents), MEMBERS)
        self.assertEqual(db.session.get(Collection, self.cid).mendeley_folder_id, "folder-1")
        linked = [paper.mendeley_doc_id for paper in self.papers]
        self.assertTrue(all(linked[:MEMBERS]))
        self.assertEqual(linked[MEMBERS:], [None, None, None])  # excluded, hidden, outsider
        # The connection is probed once per pass, not once per batch.
        self.assertEqual(self.check_connection.call_count, 1)

    def test_calls_carry_the_token_and_the_documented_media_types(self):
        self._export()

        seen = dict(zip(self.mendeley.calls, self.mendeley.headers))  # last call of each kind
        folder, document = "application/vnd.mendeley-folder.1+json", "application/vnd.mendeley-document.1+json"
        self.assertEqual(seen[("get", "/folders")], {"Authorization": "Bearer token", "Accept": folder})
        self.assertEqual(seen[("post", "/folders")], {"Authorization": "Bearer token", "Content-Type": folder})
        self.assertEqual(seen[("get", "/documents")], {"Authorization": "Bearer token", "Accept": document})
        self.assertEqual(seen[("post", "/documents")], {"Authorization": "Bearer token", "Content-Type": document})
        self.assertEqual(
            seen[("post", "/folders/folder-1/documents")], {"Authorization": "Bearer token", "Content-Type": document}
        )

    def test_a_later_export_sends_only_what_is_new(self):
        self._export()
        self.mendeley.calls.clear()

        self.assertEqual([reply["sent"] for reply in self._export()], [0])
        self.assertEqual(self._posts(), [])  # nothing new: nothing is written to Mendeley

        newcomer = _paper(100)
        db.session.add(newcomer)
        db.session.flush()
        db.session.add(PaperCollection(paper_id=newcomer.id, collection_id=self.cid))
        db.session.commit()

        (reply,) = self._export()
        self.assertEqual((reply["sent"], reply["total"]), (1, MEMBERS + 1))
        self.assertEqual(self._posts(), [("post", "/documents"), ("post", "/folders/folder-1/documents")])
        in_folder = self.mendeley.folder_documents["folder-1"]
        self.assertEqual(len(in_folder), len(set(in_folder)))  # and nothing was added twice

    def test_what_mendeley_already_holds_is_linked_not_duplicated(self):
        # A folder of that name (give or take case and spaces) and one of the papers are in
        # the library already: the document under a prefixed, versioned arXiv id, as other
        # importers write it.
        self.mendeley.folders["theirs"] = "thermal detection "
        self.mendeley.folder_documents["theirs"] = []
        held = self.members[3]
        self.mendeley.documents["theirs-doc"] = {
            "id": "theirs-doc",
            "title": held.title,
            "identifiers": {"arxiv": f"arXiv:{held.arxiv_id}v2"},
        }
        self.mendeley.documents["no-identifiers"] = {"id": "no-identifiers", "title": "A book"}

        self._export()

        self.assertEqual(list(self.mendeley.folders), ["theirs"])
        self.assertEqual(db.session.get(Collection, self.cid).mendeley_folder_id, "theirs")
        self.assertEqual(db.session.get(Paper, held.id).mendeley_doc_id, "theirs-doc")
        self.assertEqual(len(self.mendeley.documents), MEMBERS + 1)  # MEMBERS - 1 new ones, plus the two
        self.assertEqual(len(self.mendeley.folder_documents["theirs"]), MEMBERS)
        self.assertIn("theirs-doc", self.mendeley.folder_documents["theirs"])

    def test_the_folder_is_followed_by_id_and_replaced_when_deleted(self):
        self._export()

        # Renamed on both sides: still the same folder, by its remembered id.
        self.mendeley.folders["folder-1"] = "Renamed in Mendeley"
        db.session.get(Collection, self.cid).name = "Renamed here"
        db.session.commit()
        self._export()
        self.assertEqual(list(self.mendeley.folders), ["folder-1"])

        # Deleted in Mendeley (its documents stay in the library): a new folder gets them all,
        # and no document is created twice.
        del self.mendeley.folders["folder-1"], self.mendeley.folder_documents["folder-1"]
        replies = self._export()

        self.assertEqual(sum(reply["sent"] for reply in replies), MEMBERS)
        (new_folder,) = self.mendeley.folders
        self.assertNotEqual(new_folder, "folder-1")
        self.assertEqual(self.mendeley.folders[new_folder], "Renamed here")
        self.assertEqual(db.session.get(Collection, self.cid).mendeley_folder_id, new_folder)
        self.assertEqual(self._folder_arxiv_ids(new_folder), [paper.arxiv_id for paper in self.members])
        self.assertEqual(len(self.mendeley.documents), MEMBERS)

    def test_a_folder_is_not_shared_with_the_collection_that_had_its_name(self):
        self._export()
        db.session.get(Collection, self.cid).name = "Renamed here"
        twin = Collection(name="Thermal detection")  # the first collection's old name
        db.session.add(twin)
        db.session.flush()
        db.session.add(PaperCollection(paper_id=self.members[0].id, collection_id=twin.id))
        db.session.commit()
        url = f"/api/collections/{twin.id}/mendeley"

        # Mendeley still calls the first collection's folder by that name and takes no second one.
        response = self.client.post(url, json={"after": 0}, headers=self.csrf)
        self.assertEqual(response.status_code, 502)
        self.assertIn("A folder with the same name already exists", response.get_json()["error"])
        self.assertIsNone(db.session.get(Collection, twin.id).mendeley_folder_id)
        self.assertEqual(len(self.mendeley.folder_documents["folder-1"]), MEMBERS)

        # Renamed there as well, the name is free: the twin gets a folder of its own, and the
        # paper the two collections share stays one document, in both folders.
        self.mendeley.folders["folder-1"] = "Renamed here"
        reply = self.client.post(url, json={"after": 0}, headers=self.csrf).get_json()
        self.assertEqual((reply["sent"], reply["total"]), (1, 1))
        own = db.session.get(Collection, twin.id).mendeley_folder_id
        self.assertNotEqual(own, "folder-1")
        self.assertEqual(self.mendeley.folder_documents[own], [self.members[0].mendeley_doc_id])
        self.assertEqual(len(self.mendeley.documents), MEMBERS)

    def test_a_document_trashed_in_mendeley_is_left_alone(self):
        self._export()
        self.mendeley.trashed.add(self.mendeley.folder_documents["folder-1"][0])
        self.mendeley.calls.clear()

        (reply,) = self._export()

        # Still the folder's, in Mendeley's eyes: not sent again, and not reported.
        self.assertEqual((reply["sent"], reply["failures"]), (0, []))
        self.assertEqual(self._posts(), [])

    def test_a_conflict_on_adding_means_the_folder_has_it(self):
        self._export()
        # The listing misses one document that Mendeley then says is in the folder.
        self.mendeley.folder_documents["folder-1"].pop()
        self.mendeley.intercept = lambda method, path, body: (
            409 if (method, path) == ("post", "/folders/folder-1/documents") else None
        )

        (reply,) = self._export()

        self.assertEqual((reply["sent"], reply["failures"], reply["after"]), (0, [], None))

    def test_a_conflict_on_creating_is_a_failure(self):
        self.mendeley.intercept = lambda method, path, body: 409 if (method, path) == ("post", "/documents") else None

        replies = self._export()

        self.assertEqual(sum(reply["sent"] for reply in replies), 0)
        self.assertEqual(sum(len(reply["failures"]) for reply in replies), MEMBERS)

    def test_a_linked_document_that_is_gone_is_sent_again_on_the_next_run(self):
        self._export()
        paper = self.members[0]
        gone = paper.mendeley_doc_id
        # Deleted for good in Mendeley, which takes it out of the folder too.
        del self.mendeley.documents[gone]
        self.mendeley.folder_documents["folder-1"].remove(gone)

        (reply,) = self._export()

        self.assertEqual(reply["sent"], 0)
        (failure,) = reply["failures"]
        self.assertIn("404", failure)
        self.assertIn("next run", failure)
        self.assertIsNone(db.session.get(Paper, paper.id).mendeley_doc_id)

        (reply,) = self._export()
        self.assertEqual((reply["sent"], reply["failures"]), (1, []))
        self.assertEqual(self._folder_arxiv_ids("folder-1"), [member.arxiv_id for member in self.members])

    def test_long_listings_are_read_to_the_end(self):
        self.mendeley.page_size = 5  # the folder, its documents and the library all span pages
        for i in range(7):
            self.mendeley.folders[f"other-{i}"] = f"Other {i}"
            self.mendeley.folder_documents[f"other-{i}"] = []
        self.mendeley.folders["theirs"] = "Thermal detection"  # on the second page of folders
        self.mendeley.folder_documents["theirs"] = []
        self._export()
        self.mendeley.calls.clear()

        newcomer = _paper(100)
        db.session.add(newcomer)
        db.session.flush()
        db.session.add(PaperCollection(paper_id=newcomer.id, collection_id=self.cid))
        db.session.commit()
        # The newcomer is already in the library, on its last page.
        self.mendeley.documents["late"] = {"id": "late", "identifiers": {"arxiv": newcomer.arxiv_id}}

        (reply,) = self._export()

        self.assertEqual(len(self.mendeley.folders), 8)  # found on page two, not created again
        self.assertEqual(reply["sent"], 1)  # every earlier paper was seen in the folder's later pages
        self.assertEqual(self._posts(), [("post", "/folders/theirs/documents")])
        self.assertEqual(db.session.get(Paper, newcomer.id).mendeley_doc_id, "late")

    # ── failures ──

    def test_a_rejected_paper_is_reported_and_neither_blocks_nor_loops(self):
        bad = self.members[0]
        self.mendeley.intercept = lambda method, path, body: (
            400 if (method, path) == ("post", "/documents") and body["title"] == bad.title else None
        )

        replies = self._export()  # ends, although the first paper of the pass keeps failing

        failures = [failure for reply in replies for failure in reply["failures"]]
        self.assertEqual(len(failures), 1)
        self.assertIn(bad.title, failures[0])
        self.assertIn("400", failures[0])
        self.assertIn("intercepted", failures[0])  # Mendeley's own explanation
        self.assertEqual(sum(reply["sent"] for reply in replies), MEMBERS - 1)
        self.assertIsNone(db.session.get(Paper, bad.id).mendeley_doc_id)

        self.mendeley.intercept = lambda method, path, body: None
        (reply,) = self._export()  # the next pass retries it
        self.assertEqual((reply["sent"], reply["failures"]), (1, []))
        self.assertEqual(len(self.mendeley.folder_documents["folder-1"]), MEMBERS)

    def test_one_paper_that_draws_a_server_error_does_not_block_the_rest(self):
        bad = self.members[0]
        self.mendeley.intercept = lambda method, path, body: (
            500 if (method, path) == ("post", "/documents") and body["title"] == bad.title else None
        )

        replies = self._export()

        (failure,) = [failure for reply in replies for failure in reply["failures"]]
        self.assertIn(bad.title, failure)
        self.assertIn("500", failure)
        self.assertEqual(sum(reply["sent"] for reply in replies), MEMBERS - 1)

    def test_an_outage_stops_the_pass_and_the_next_one_resumes_without_duplicates(self):
        # Three papers are in the folder when Mendeley starts answering 503 to every addition.
        def outage(method, path, body):
            adding = method == "post" and path.endswith("/folder-1/documents")
            return 503 if adding and len(self.mendeley.folder_documents["folder-1"]) == 3 else None

        self.mendeley.intercept = outage
        response = self._post()

        # The first 503 is put down to its paper, the second in a row to Mendeley: the pass stops.
        self.assertEqual(response.status_code, 502)
        self.assertIn("503", response.get_json()["error"])
        self.assertIn(self.members[4].title, response.get_json()["error"])
        self.assertEqual(response.get_json()["sent"], 3)
        self.assertEqual(self._posts()[-1], ("post", "/folders/folder-1/documents"))
        self.assertEqual(self._posts().count(("post", "/folders/folder-1/documents")), 5)
        self.assertEqual(len(self.mendeley.documents), 5)
        # What was done is kept, the half-sent papers' document ids included.
        self.assertEqual(sum(1 for paper in self.members if paper.mendeley_doc_id), 5)

        self.mendeley.intercept = lambda method, path, body: None
        replies = self._export()

        self.assertEqual(sum(reply["sent"] for reply in replies), MEMBERS - 3)
        self.assertEqual(len(self.mendeley.documents), MEMBERS)
        self.assertEqual(self._folder_arxiv_ids("folder-1"), [paper.arxiv_id for paper in self.members])

    def test_rate_limiting_stops_at_the_first_refusal(self):
        self.mendeley.intercept = lambda method, path, body: 429 if (method, path) == ("post", "/documents") else None

        response = self._post()

        self.assertEqual(response.status_code, 502)
        self.assertIn("429", response.get_json()["error"])
        self.assertEqual(self._posts().count(("post", "/documents")), 1)

    def test_a_dead_refresh_token_stops_instead_of_failing_every_paper(self):
        self.mendeley.intercept = lambda method, path, body: 401 if (method, path) == ("post", "/documents") else None
        refused = requests.Response()
        refused.status_code = 400  # the token endpoint's invalid_grant
        with patch(
            "app.services.mendeley.MendeleyClient._refresh_access_token",
            side_effect=requests.HTTPError("400 Client Error", response=refused),
        ):
            response = self._post()

        self.assertEqual(response.status_code, 502)
        self.assertIn("could not be refreshed", response.get_json()["error"])
        self.assertEqual(self._posts().count(("post", "/documents")), 1)

    def test_a_lost_reply_does_not_leave_a_duplicate_behind(self):
        # Mendeley creates the second document, but its answer never arrives.
        real_post, lost = self.mendeley.post, []

        def flaky_post(url, **kwargs):
            response = real_post(url, **kwargs)
            if url == f"{FakeMendeley.BASE}/documents" and len(self.mendeley.documents) == 2 and not lost:
                lost.append(url)
                raise requests.Timeout("read timed out")
            return response

        with patch("app.services.mendeley.requests.post", flaky_post):
            self.assertEqual(self._post().status_code, 502)
            self.assertEqual(len(self.mendeley.documents), 2)
            self.assertEqual(sum(1 for paper in self.members if paper.mendeley_doc_id), 1)

            self._export()

        self.assertEqual(len(self.mendeley.documents), MEMBERS)
        self.assertEqual(self._folder_arxiv_ids("folder-1"), [paper.arxiv_id for paper in self.members])

    def test_a_pagination_link_off_the_api_is_not_followed(self):
        def get(url, **kwargs):
            response = self.mendeley.get(url, **kwargs)
            response.headers["Link"] = '<https://evil.example/folders?marker=1>; rel="next"'
            return response

        with patch("app.services.mendeley.requests.get", get):
            response = self._post()

        self.assertEqual(response.status_code, 502)
        self.assertIn("leaves the API", response.get_json()["error"])
        self.assertEqual(self.mendeley.calls, [("get", "/folders")])  # the token went nowhere else

    # ── next to the per-paper send ──

    def test_a_card_sent_after_its_collection_makes_no_second_document(self):
        self._export()
        paper = self.members[0]
        self.mendeley.calls.clear()

        # The card's button was drawn before the export: the page still offers to send it.
        response = self.client.post(f"/api/papers/{paper.id}/mendeley", json={}, headers=self.csrf)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["document_id"], paper.mendeley_doc_id)
        self.assertEqual(self.mendeley.calls, [])
        self.assertEqual(len(self.mendeley.documents), MEMBERS)

    def test_a_card_clicked_during_a_batch_waits_for_the_batch(self):
        target_id, target_arxiv_id = self.members[0].id, self.members[0].arxiv_id
        real_post, clicked = self.mendeley.post, {}

        def click():  # another tab: a request with an app context and a database session of its own
            client = self.app.test_client()
            client.get("/")
            with client.session_transaction() as session:
                headers = {"X-CSRF-Token": session["settings_csrf_token"]}
            clicked["response"] = client.post(f"/api/papers/{target_id}/mendeley", json={}, headers=headers)

        thread = threading.Thread(target=click)

        def post(url, **kwargs):
            # While Mendeley is creating the first paper's document, that paper's card is clicked.
            if url == f"{FakeMendeley.BASE}/documents" and not thread.ident:
                thread.start()
                thread.join(0.5)  # time enough for the whole click, were nothing holding it back
            return real_post(url, **kwargs)

        with patch("app.services.mendeley.requests.post", post):
            self.assertEqual(self._post().status_code, 200)
            thread.join(20)

        self.assertEqual(clicked["response"].get_json()["message"], "Already in Mendeley.")
        arxiv_ids = [document["identifiers"]["arxiv"] for document in self.mendeley.documents.values()]
        self.assertEqual(arxiv_ids.count(target_arxiv_id), 1)

    # ── the endpoint ──

    def test_endpoint_guards(self):
        self.assertEqual(self.client.post(f"/api/collections/{self.cid}/mendeley", json={}).status_code, 400)  # CSRF
        for bad in ("5", -1, True, None, 1.5):
            self.assertEqual(self._post(bad).status_code, 400, bad)
        self.assertEqual(
            self.client.post("/api/collections/9999/mendeley", json={}, headers=self.csrf).status_code, 404
        )

        self.check_connection.return_value = {"status": "expired", "message": "Mendeley token expired."}
        response = self._post()
        self.assertEqual(response.status_code, 400)
        self.assertIn("Mendeley not connected", response.get_json()["error"])
        self.assertEqual(self.mendeley.calls, [])

    def test_the_button_shows_only_in_a_collection_and_only_when_connected(self):
        self.assertIn(
            'id="collection-mendeley-btn"', self.client.get(f"/?collection={self.cid}").get_data(as_text=True)
        )
        self.assertNotIn("collection-mendeley-btn", self.client.get("/").get_data(as_text=True))

        self.app.extensions.pop("mendeley_status_cache")
        self.check_connection.return_value = {"status": "no_token", "message": "Mendeley not authorized."}
        self.assertNotIn(
            'id="collection-mendeley-btn"', self.client.get(f"/?collection={self.cid}").get_data(as_text=True)
        )

    # ── schema ──

    def test_ensure_schema_adds_the_folder_id_to_a_legacy_db(self):
        from app.schema import ensure_schema

        db.session.execute(text("ALTER TABLE collections DROP COLUMN mendeley_folder_id"))
        db.session.commit()

        ensure_schema()
        ensure_schema()  # idempotent

        self.assertIn(
            "mendeley_folder_id", {column["name"] for column in inspect(db.engine).get_columns("collections")}
        )
        self.assertIsNone(db.session.get(Collection, self.cid).mendeley_folder_id)
