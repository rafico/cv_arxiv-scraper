from flask import Blueprint, render_template, request
from app.models import Paper, db

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    query = Paper.query

    # Filter by match type (uses LIKE since match_type can be "Author + Affiliation")
    match_type = request.args.get("match_type")
    if match_type:
        query = query.filter(Paper.match_type.contains(match_type))

    # Filter by scrape date
    date = request.args.get("date")
    if date:
        query = query.filter(Paper.scraped_date == date)

    # Search
    q = request.args.get("q", "").strip()
    if q:
        search = f"%{q}%"
        query = query.filter(
            db.or_(
                Paper.title.ilike(search),
                Paper.authors.ilike(search),
                Paper.matched_terms.ilike(search),
            )
        )

    # Sort: newest first, then by match priority
    # Papers containing "Author" sort first, then "Affiliation", then "Title"
    query = query.order_by(
        Paper.scraped_date.desc(),
        db.case(
            (Paper.match_type.contains("Author"), 1),
            (Paper.match_type.contains("Affiliation"), 2),
            else_=3,
        ),
    )

    papers = query.all()

    # Count papers by match type (handles compound types like "Author + Affiliation")
    type_counts = {"Author": 0, "Affiliation": 0, "Title": 0}
    for p in papers:
        for t in type_counts:
            if t in p.match_type:
                type_counts[t] += 1

    # Get distinct scrape dates for filter dropdown
    dates = (
        db.session.query(Paper.scraped_date)
        .distinct()
        .order_by(Paper.scraped_date.desc())
        .all()
    )
    dates = [d[0] for d in dates]

    return render_template(
        "dashboard.html",
        papers=papers,
        dates=dates,
        type_counts=type_counts,
        current_filters={
            "match_type": match_type,
            "date": date,
            "q": q,
        },
    )
