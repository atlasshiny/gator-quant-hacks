# Databento MBP-10 Parquet Schema Specification (GLBX.MDP3 / CME Globex)

## 1. Core Event Metadata (13 Columns)

| Field Name | Databento C Type | Pandas / cuDF Type | Description |
| :--- | :--- | :--- | :--- |
| `ts_event` | `uint64_t` | `datetime64[ns, UTC]` | Matching engine timestamp (DataFrame Index) |
| `ts_recv` | `uint64_t` | `datetime64[ns, UTC]` | Capture-server receipt timestamp |
| `rtype` | `uint8_t` | `uint8` | Record type sentinel value (`10` for `mbp-10`) |
| `publisher_id` | `uint16_t` | `uint16` | Dataset/Venue identifier (CME Globex) |
| `instrument_id` | `uint32_t` | `uint32` | CME numeric security identifier |
| `action` | `char` | `category` / `object` | Event action: `'A'` (Add), `'C'` (Cancel), `'M'` (Modify), `'R'` (Clear), `'T'` (Trade) |
| `side` | `char` | `category` / `object` | Aggressor side: `'A'` (Ask/Sell), `'B'` (Bid/Buy), `'N'` (None) |
| `depth` | `uint8_t` | `uint8` | Book depth level where event occurred |
| `price` | `int64_t` | `float64` | Order/Trade price (converted from $10^{-9}$ fixed-point precision) |
| `size` | `uint32_t` | `uint32` | Order/Trade quantity |
| `flags` | `uint8_t` | `uint8` | Bit field indicating message characteristics/data quality |
| `ts_in_delta` | `int32_t` | `int32` | Matching engine send delta relative to `ts_recv` (in nanoseconds) |
| `sequence` | `uint32_t` | `uint32` | Venue message sequence number |

---

## 2. Level 2 Order Book Depth Pattern (60 Columns)

| Field Pattern | Databento C Type | Pandas / cuDF Type | Description |
| :--- | :--- | :--- | :--- |
| `bid_px_00` ... `bid_px_09` | `int64_t` | `float64` | Bid price at depth level $N$ ($00 = \text{L1}$, $09 = \text{L10}$) |
| `ask_px_00` ... `ask_px_09` | `int64_t` | `float64` | Ask price at depth level $N$ ($00 = \text{L1}$, $09 = \text{L10}$) |
| `bid_sz_00` ... `bid_sz_09` | `uint32_t` | `uint32` | Aggregate bid quantity at depth level $N$ |
| `ask_sz_00` ... `ask_sz_09` | `uint32_t` | `uint32` | Aggregate ask quantity at depth level $N$ |
| `bid_ct_00` ... `bid_ct_09` | `uint32_t` | `uint32` | Number of distinct bid orders at depth level $N$ |
| `ask_ct_00` ... `ask_ct_09` | `uint32_t` | `uint32` | Number of distinct ask orders at depth level $N$ |

---

## 3. Level-by-Level Column Matrix

| Depth Level $N$ | Bid Price | Ask Price | Bid Size | Ask Size | Bid Count | Ask Count |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **00 (L1)** | `bid_px_00` | `ask_px_00` | `bid_sz_00` | `ask_sz_00` | `bid_ct_00` | `ask_ct_00` |
| **01 (L2)** | `bid_px_01` | `ask_px_01` | `bid_sz_01` | `ask_sz_01` | `bid_ct_01` | `ask_ct_01` |
| **02 (L3)** | `bid_px_02` | `ask_px_02` | `bid_sz_02` | `ask_sz_02` | `bid_ct_02` | `ask_ct_02` |
| **03 (L4)** | `bid_px_03` | `ask_px_03` | `bid_sz_03` | `ask_sz_03` | `bid_ct_03` | `ask_ct_03` |
| **04 (L5)** | `bid_px_04` | `ask_px_04` | `bid_sz_04` | `ask_sz_04` | `bid_ct_04` | `ask_ct_04` |
| **05 (L6)** | `bid_px_05` | `ask_px_05` | `bid_sz_05` | `ask_sz_05` | `bid_ct_05` | `ask_ct_05` |
| **06 (L7)** | `bid_px_06` | `ask_px_06` | `bid_sz_06` | `ask_sz_06` | `bid_ct_06` | `ask_ct_06` |
| **07 (L8)** | `bid_px_07` | `ask_px_07` | `bid_sz_07` | `ask_sz_07` | `bid_ct_07` | `ask_ct_07` |
| **08 (L9)** | `bid_px_08` | `ask_px_08` | `bid_sz_08` | `ask_sz_08` | `bid_ct_08` | `ask_ct_08` |
| **09 (L10)** | `bid_px_09` | `ask_px_09` | `bid_sz_09` | `ask_sz_09` | `bid_ct_09` | `ask_ct_09` |