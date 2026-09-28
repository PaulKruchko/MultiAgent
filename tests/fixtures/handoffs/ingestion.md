## Summary

TLSF gives O(1) malloc/free with bounded fragmentation.

## Sources

- Masmano et al., "TLSF: a New Dynamic Memory Allocator for Real-Time Systems" (2004)

## Key Facts

- TLSF uses two bitmap levels and `ffs`/`fls` instructions for O(1) search.

## Data Tables

| Design | malloc | free |
|---|---|---|
| TLSF | O(1) | O(1) |

## Open Questions

None.
