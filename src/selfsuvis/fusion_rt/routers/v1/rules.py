"""Fusion rules CRUD."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from selfsuvis.fusion_rt.db import get_db_pool
from selfsuvis.fusion_rt.deps import require_api_key
from selfsuvis.fusion_rt.routers.v1.schemas import RuleCreate, RuleResponse, RuleUpdate
from selfsuvis.pipeline.core import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


def _row_to_rule(row) -> RuleResponse:
    return RuleResponse(
        rule_id=row["rule_id"],
        label=row["label"],
        modalities=list(row["modalities"]),
        zone_id=row["zone_id"],
        window_s=row["window_s"],
        min_confidence=row["min_confidence"],
        enabled=row["enabled"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


@router.get("/rules", summary="List rules")
async def list_rules(request: Request) -> dict:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM fusion_rules ORDER BY created_at DESC")
    return {"rules": [_row_to_rule(r).model_dump() for r in rows]}


@router.post("/rules", status_code=201, summary="Create rule")
async def create_rule(request: Request, body: RuleCreate) -> RuleResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        existing = await conn.fetchval(
            "SELECT rule_id FROM fusion_rules WHERE rule_id = $1", body.rule_id
        )
        if existing:
            raise HTTPException(status_code=409, detail="Rule already exists")
        row = await conn.fetchrow(
            """
            INSERT INTO fusion_rules
                (rule_id, label, modalities, zone_id, window_s, min_confidence, enabled)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            body.rule_id,
            body.label,
            body.modalities,
            body.zone_id,
            body.window_s,
            body.min_confidence,
            body.enabled,
        )
    return _row_to_rule(row)


@router.get("/rules/{rule_id}", summary="Get rule")
async def get_rule(request: Request, rule_id: str) -> RuleResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM fusion_rules WHERE rule_id = $1", rule_id)
    if not row:
        raise HTTPException(status_code=404, detail="Rule not found")
    return _row_to_rule(row)


@router.put("/rules/{rule_id}", summary="Update rule")
async def update_rule(request: Request, rule_id: str, body: RuleUpdate) -> RuleResponse:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT * FROM fusion_rules WHERE rule_id = $1", rule_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Rule not found")

        updates = body.model_dump(exclude_none=True)
        if not updates:
            return _row_to_rule(existing)

        updates["updated_at"] = datetime.now(timezone.utc)
        set_clauses = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(updates))
        values = [rule_id] + list(updates.values())
        row = await conn.fetchrow(
            f"UPDATE fusion_rules SET {set_clauses} WHERE rule_id = $1 RETURNING *",
            *values,
        )
    return _row_to_rule(row)


@router.delete("/rules/{rule_id}", status_code=204, summary="Delete rule")
async def delete_rule(request: Request, rule_id: str) -> None:
    pool = get_db_pool(request)
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM fusion_rules WHERE rule_id = $1", rule_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Rule not found")
