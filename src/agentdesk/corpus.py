"""Sample corpus.

Small, synthetic, and deterministic so CI has no network dependency and tests
assert on exact behaviour. Real use points the retriever at a real document
store -- the Document interface is the only contract.
"""

from agentdesk.agents.roles import Document

SAMPLE_CORPUS = [
    Document(
        doc_id="acme-2023-mda",
        source="ACME FY2023 10-K Item 7",
        text=(
            "Consolidated revenue for fiscal 2023 was $4.82 billion, an increase "
            "of 7.3% compared to $4.49 billion in fiscal 2022. Gross margin "
            "declined to 26.4% from 28.1%, a decrease of 170 basis points."
        ),
    ),
    Document(
        doc_id="acme-2023-drivers",
        source="ACME FY2023 10-K Item 7",
        text=(
            "The margin decline reflects higher raw material costs, incremental "
            "tariff expense of approximately $34 million, and unfavorable "
            "manufacturing absorption at two facilities during the third quarter."
        ),
    ),
    Document(
        doc_id="acme-2023-risk",
        source="ACME FY2023 10-K Item 1A",
        text=(
            "A substantial portion of the titanium and nickel alloys used in our "
            "aerospace products is sourced from a limited number of qualified "
            "suppliers. Qualifying an alternative supplier requires twelve to "
            "eighteen months and customer approval."
        ),
    ),
    Document(
        doc_id="acme-2023-cash",
        source="ACME FY2023 10-K Item 7",
        text=(
            "Operating cash flow was $612 million compared to $701 million in the "
            "prior year, primarily attributable to an increase in working capital "
            "as the Company built inventory of long-lead-time alloys."
        ),
    ),
]
