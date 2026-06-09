from django.db import connection


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    jurisdiction: str | None = None,
    exclude_cross_references: bool = True,
    limit: int = 8,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Run hybrid pgvector + FTS search with RRF fusion.

    Returns a list of dicts with record fields and rrf_score.
    """
    params = [
        query_embedding,   # dense ORDER BY (CTE)
        query_embedding,   # dense ORDER BY (inner)
        query_text,        # sparse ts_rank
        query_text,        # sparse plainto_tsquery (WHERE)
        rrf_k,             # RRF constant (dense)
        rrf_k,             # RRF constant (sparse)
        limit,
    ]

    jurisdiction_filter = ""
    cross_ref_filter = ""

    if jurisdiction:
        jurisdiction_filter = "AND sd.jurisdiction = %s"
        # SQL param order: [emb(ROW_NUMBER), jur(dense WHERE), emb(ORDER BY),
        #                   qt(ts_rank), qt(plainto_tsquery WHERE), jur(sparse WHERE),
        #                   rrf_k, rrf_k, limit]
        params.insert(1, jurisdiction)  # after first embedding (dense ROW_NUMBER)
        params.insert(5, jurisdiction)  # after both query_texts (sparse WHERE)

    if exclude_cross_references:
        cross_ref_filter = "AND r.is_cross_reference = FALSE"

    sql = f"""
    WITH dense AS (
        SELECT
            r.id,
            ROW_NUMBER() OVER (ORDER BY r.embedding <=> %s::vector) AS rank
        FROM records_retentionrecord r
        JOIN records_sourcedocument sd ON r.source_document_id = sd.id
        WHERE r.embedding IS NOT NULL
          {jurisdiction_filter}
          {cross_ref_filter}
        ORDER BY r.embedding <=> %s::vector
        LIMIT 60
    ),
    sparse AS (
        SELECT
            id,
            ROW_NUMBER() OVER (ORDER BY rank_score DESC) AS rank
        FROM (
            SELECT
                r.id,
                ts_rank(r.search_vector, plainto_tsquery('english', %s)) AS rank_score
            FROM records_retentionrecord r
            JOIN records_sourcedocument sd ON r.source_document_id = sd.id
            WHERE r.search_vector @@ to_tsquery('english',
                      replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')
                  )
              {jurisdiction_filter}
              {cross_ref_filter}
            ORDER BY rank_score DESC
            LIMIT 60
        ) sub
    ),
    fused AS (
        SELECT
            COALESCE(d.id, s.id) AS id,
            COALESCE(1.0 / (%s + d.rank), 0) AS dense_score,
            COALESCE(1.0 / (%s + s.rank), 0) AS sparse_score
        FROM dense d
        FULL OUTER JOIN sparse s ON d.id = s.id
    )
    SELECT
        r.id,
        r.record_number,
        r.record_title,
        r.record_description,
        r.minimum_retention_period,
        r.custodian_requirement,
        r.regulatory_citations,
        r.page_number,
        r.is_permanent,
        sd.document_title,
        sd.jurisdiction,
        sd.entity_type,
        sd.documentcloud_url,
        f.dense_score,
        f.sparse_score,
        f.dense_score + f.sparse_score AS rrf_score
    FROM fused f
    JOIN records_retentionrecord r ON f.id = r.id
    JOIN records_sourcedocument sd ON r.source_document_id = sd.id
    ORDER BY rrf_score DESC
    LIMIT %s
    """

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def document_search(
    query_embedding: list[float],
    jurisdiction: str | None = None,
    limit: int = 6,
) -> list[dict]:
    """
    Pure vector similarity search over DocumentChunks.
    Returns chunks ordered by cosine distance (nearest first).
    """
    jurisdiction_filter = "AND sd.jurisdiction = %s" if jurisdiction else ""
    sql = f"""
    SELECT
        c.id,
        c.chunk_index,
        c.page_number,
        c.text,
        c.token_count,
        sd.document_title,
        sd.document_type,
        sd.jurisdiction,
        sd.documentcloud_url,
        1 - (c.embedding <=> %s::vector) AS similarity_score
    FROM records_documentchunk c
    JOIN records_supportingdocument sd ON c.supporting_document_id = sd.id
    WHERE c.embedding IS NOT NULL
      {jurisdiction_filter}
    ORDER BY c.embedding <=> %s::vector ASC
    LIMIT %s
    """
    params = [query_embedding]
    if jurisdiction:
        params.append(jurisdiction)
    params += [query_embedding, limit]
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
