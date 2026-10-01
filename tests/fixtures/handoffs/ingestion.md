## Summary

TLSF gives O(1) malloc/free with bounded fragmentation.

## Sources

- [S1] TLSF: a New Dynamic Memory Allocator for Real-Time Systems
  - Authors: M. Masmano; I. Ripoll; A. Crespo; J. Real
  - Venue: Proceedings of the 16th Euromicro Conference on Real-Time Systems (ECRTS 2004), pp. 79-86
  - Year: 2004
  - DOI: 10.1109/EMRTS.2004.1311009
  - Excerpt (Section 3, p. 81): "a two-level segregated fit where bitmaps locate a suitable free list in constant time"

## Key Facts

- TLSF uses two bitmap levels and `ffs`/`fls` instructions for O(1) search [S1].

## Data Tables

| Design | malloc | free |
|---|---|---|
| TLSF | O(1) | O(1) |

## Open Questions

None.
