from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request, Response

from stock_calculator.ranking import rank_candidates, render_rank_result


SUPPORTED_FORMATS = {"table", "csv", "json"}

app = FastAPI(title="rTrade Insights Local API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rank")
async def rank(
    request: Request,
    format: str = Query(default="table", pattern="^(table|csv|json)$"),
) -> Response:
    body = (await request.body()).decode("utf-8")
    try:
        rendered = render_rank_result(rank_candidates(body), format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=rendered, media_type=_media_type(format))


def _media_type(output_format: str) -> str:
    return {
        "table": "text/plain",
        "csv": "text/csv",
        "json": "application/json",
    }[output_format]
