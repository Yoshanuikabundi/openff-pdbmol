from typing import Iterable, Iterator, TypeVar

from openff.units import Quantity, unit

__all__ = [
    "unwrap",
    "flatten",
    "float_or_unknown",
    "dec_hex",
    "cryst_to_box_vectors",
    "__UNSET__",
]

T = TypeVar("T")
U = TypeVar("U")


class __UNSET__:
    pass


def unwrap(container: Iterable[T], msg: str = "") -> T:
    """
    Unwrap an iterable only if it has a single element; raise ValueError otherwise
    """
    if msg:
        msg += ": "

    iterator = iter(container)

    try:
        value = next(iterator)
    except StopIteration:
        raise ValueError(msg + "container has no elements")

    try:
        next(iterator)
    except StopIteration:
        return value

    raise ValueError(msg + "container has multiple elements")


def flatten(container: Iterable[Iterable[T]]) -> Iterator[T]:
    for inner in container:
        yield from inner


def with_neighbours(
    iterable: Iterable[T], default: U = None
) -> Iterator[tuple[T | U, T, T | U]]:
    iterator = iter(iterable)

    pred: T | U = default
    this: T
    succ: T | U

    try:
        this = next(iterator)
    except StopIteration:
        return

    for succ in iterator:
        yield (pred, this, succ)
        pred = this
        this = succ

    succ = default
    yield (pred, this, succ)


def float_or_unknown(s: str) -> float | None:
    if s == "?":
        return None
    return float(s)


def dec_hex(s: str) -> int:
    """
    Interpret a string as a decimal or hexadecimal integer.

    For a string of length n, the string is interpreted as decimal if the value
    is < 10^n. This makes the dec_hex representation identical to a decimal
    integer, except for strings that cannot be parsed as a decimal. For these
    strings, the first hexadecimal number is interpreted as 10^n, and subsequent
    numbers continue from there. For example, in PDB files, a fixed width column
    format, residue numbers for large systems follow this representation:

        "   1" -> 1
        "   2" -> 2
        ...
        "9999" -> 9999
        "A000" -> 10000
        "A001" -> 10001
        ...
        "A009" -> 10009
        "A00A" -> 10010
        "A00B" -> 10011
        ...
        "A00F" -> 10015
        "A010" -> 10016
        ...
    """

    try:
        return int(s, 10)
    except ValueError:
        n = len(s)
        parsed_as_hex = int(s, 16)
        smallest_hex = 0xA * 16 ** (n - 1)
        largest_dec = 10**n - 1
        return parsed_as_hex - smallest_hex + largest_dec + 1


def cryst_to_box_vectors(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> Quantity:  # type: ignore[invalid-type]
    import openmm.unit
    from openmm.app.internal.unitcell import computePeriodicBoxVectors
    from openmm.unit import (
        nanometer as openmm_unit_nanometer,  # type: ignore[import-not-found]
    )

    box_vectors = computePeriodicBoxVectors(
        openmm.unit.Quantity(a, openmm.unit.angstrom),
        openmm.unit.Quantity(b, openmm.unit.angstrom),
        openmm.unit.Quantity(c, openmm.unit.angstrom),
        openmm.unit.Quantity(alpha, openmm.unit.degree),
        openmm.unit.Quantity(beta, openmm.unit.degree),
        openmm.unit.Quantity(gamma, openmm.unit.degree),
    )
    return box_vectors.value_in_unit(openmm_unit_nanometer) * unit.nanometer
