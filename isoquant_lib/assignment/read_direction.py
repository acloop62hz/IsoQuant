"""Protocol/external read direction shared by assignment and construction.

`.` means unavailable (legacy handling); `!` means contradictory evidence and
must never silently fall back to an annotation or splice-motif strand.
"""

CONFLICT = "!"


def resolve_read_direction(params, mapped_strand, external_strand=None):
    mode = getattr(params, "stranded", None)
    bam = "."
    if mapped_strand in ("+", "-"):
        if mode == "forward":
            bam = mapped_strand
        elif mode == "reverse":
            bam = "-" if mapped_strand == "+" else "+"
    if external_strand == CONFLICT:
        return CONFLICT
    if external_strand in ("+", "-"):
        return CONFLICT if bam in ("+", "-") and bam != external_strand else external_strand
    return bam


def external_read_direction(read):
    info = getattr(read, "additional_info", {})
    external = info.get("external_read_strand")
    terminal = info.get("pt_terminal")
    pt = {1: "+", 2: "-"}.get(terminal[0]) if terminal else None
    if external == CONFLICT or (external in ("+", "-") and pt and external != pt):
        return CONFLICT
    return pt or external


def trusted_read_direction(read, params):
    return resolve_read_direction(params, getattr(read, "mapped_strand", "."),
                                  external_read_direction(read))


def direction_compatible(direction, strand):
    return direction == "." or (direction in ("+", "-") and direction == strand)


def read_matches_strand(read, strand, params):
    return direction_compatible(trusted_read_direction(read, params), strand)


def effective_read_strand(read, params):
    direction = trusted_read_direction(read, params)
    return getattr(read, "strand", ".") if direction == "." else direction
