"""Check literal field ownership and existing parser buffer behavior."""

from __future__ import annotations

# Tests follow the native namespace layout and exercise the private parser.
# ruff: noqa: INP001, S101, SLF001
import pytest
from typing_extensions import override

from hpack import Decoder, Encoder, HeaderTuple, NeverIndexedHeaderTuple
from hpack.exceptions import HPACKDecodingError, InvalidTableIndex
from hpack.table import HeaderTable

LITERAL = b"\x40\x01k\x02vv"


class _CustomBytes(bytes):
    @override
    def __bytes__(self) -> bytes:
        return b"not the buffer contents"


class _MutatingTable(HeaderTable):
    """Change the caller's buffer during indexed-name lookup."""

    def __init__(self, backing: bytearray) -> None:
        super().__init__()
        self.backing = backing

    @override
    def get_by_index(self, index: int) -> tuple[bytes, bytes]:
        result = super().get_by_index(index)
        self.backing[-1] = ord("z")
        return result


class _ReleasingTable(HeaderTable):
    """Release the caller's view during indexed-name lookup."""

    def __init__(self, view: memoryview) -> None:
        super().__init__()
        self.view = view

    @override
    def get_by_index(self, index: int) -> tuple[bytes, bytes]:
        result = super().get_by_index(index)
        self.view.release()
        return result


def _strided_view(data: bytes) -> memoryview:
    """Keep the encoded bytes in alternate positions of immutable storage."""
    padded = bytearray(len(data) * 2)
    padded[::2] = data
    return memoryview(bytes(padded))[::2]


@pytest.mark.parametrize("huffman", [False, True])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("header_type", [HeaderTuple, NeverIndexedHeaderTuple])
def test_literal_fields_own_bytes(
    *,
    huffman: bool,
    raw: bool,
    header_type: type[HeaderTuple],
) -> None:
    """Literal and indexed names preserve field types and table ownership."""
    headers = [
        header_type(b"custom-key", b"custom-value"),
        header_type(b":path", b"/sample/path"),
    ]
    encoded = Encoder().encode(headers, huffman=huffman)
    decoder = Decoder()

    decoded = list(decoder.decode(encoded, raw=raw))
    expected = headers if raw else [(name.decode(), value.decode()) for name, value in headers]
    assert decoded == expected
    assert all(type(header) is header_type for header in decoded)

    field_type = bytes if raw else str
    assert all(type(field) is field_type for header in decoded for field in header)
    entries = decoder.header_table.dynamic_entries
    if header_type is NeverIndexedHeaderTuple:
        assert not entries
    else:
        assert len(entries) == len(headers)
        assert all(type(field) is bytes for entry in entries for field in entry)


@pytest.mark.parametrize(
    "buffer",
    [
        pytest.param(LITERAL, id="bytes"),
        pytest.param(memoryview(LITERAL), id="byte-view"),
        pytest.param(memoryview(bytearray(LITERAL)), id="mutable-view"),
        pytest.param(_strided_view(LITERAL), id="strided-view"),
        pytest.param(memoryview(LITERAL).cast("H"), id="wide-view"),
        pytest.param(memoryview(LITERAL).cast("B", (2, 3)), id="multidimensional-view"),
        pytest.param(_CustomBytes(LITERAL), id="bytes-subclass"),
        pytest.param(memoryview(_CustomBytes(LITERAL)), id="bytes-subclass-view"),
    ],
)
def test_literal_buffer_layouts(buffer: bytes | memoryview) -> None:
    """Normalization preserves the parser's accepted byte-buffer layouts."""
    decoder = Decoder()

    header, consumed = decoder._decode_literal_index(buffer)

    assert header == (b"k", b"vv")
    assert consumed == len(LITERAL)
    assert all(type(field) is bytes for field in header)
    assert list(decoder.header_table.dynamic_entries) == [(b"k", b"vv")]
    assert all(type(field) is bytes for entry in decoder.header_table.dynamic_entries for field in entry)


def test_readonly_mutable_buffer_is_snapshotted() -> None:
    """A readonly view must not expose later changes in its mutable backing."""
    backing = bytearray(b"\x44\x01v")
    decoder = Decoder()
    decoder.header_table = _MutatingTable(backing)

    # The public annotation is bytes; this deliberate buffer input checks existing runtime compatibility.
    headers = list(decoder.decode(memoryview(backing).toreadonly(), raw=True))  # ty: ignore[invalid-argument-type]

    assert backing == b"\x44\x01z"
    assert headers == [(b":path", b"v")]
    assert list(decoder.header_table.dynamic_entries) == [(b":path", b"v")]


def test_invalid_literal_index_precedes_missing_value() -> None:
    """Invalid indexed names fail before the parser reads a missing value."""
    decoder = Decoder()

    with pytest.raises(InvalidTableIndex):
        decoder.decode(b"\x7f\x7f", raw=True)

    assert not decoder.header_table.dynamic_entries


def test_literal_parser_owns_its_view_handle() -> None:
    """Releasing the caller's view does not interrupt literal parsing."""
    view = memoryview(b"\x44\x01v")
    decoder = Decoder()
    decoder.header_table = _ReleasingTable(view)

    header, consumed = decoder._decode_literal_index(view)

    assert header == (b":path", b"v")
    assert consumed == 3
    assert list(decoder.header_table.dynamic_entries) == [(b":path", b"v")]
    with pytest.raises(ValueError, match="released memoryview"):
        view.tobytes()


@pytest.mark.parametrize(
    ("data", "diagnostic_data"),
    [
        (b"\x40", b""),
        (b"\x40\x7f\x80", b"\x7f\x80"),
        (b"\x44", b""),
        (b"\x44\x7f\x80", b"\x7f\x80"),
        (b"\x7f\x80", b"\x7f\x80"),
    ],
)
def test_incomplete_literal_integer_keeps_byte_diagnostics(data: bytes, diagnostic_data: bytes) -> None:
    """Malformed literal lengths and indices retain their content and error chain."""
    decoder = Decoder()

    with pytest.raises(HPACKDecodingError) as error:
        decoder.decode(data, raw=True)

    assert str(error.value) == f"Unable to decode HPACK integer representation from {diagnostic_data!r}"
    context = error.value.__context__
    assert isinstance(context, IndexError)
    assert context.__context__ is None
    assert not decoder.header_table.dynamic_entries


def test_literal_integer_limit_keeps_byte_diagnostics() -> None:
    """Integer limit errors retain encoded byte content."""
    data = b"\x40\x7f" + b"\x80" * 5

    with pytest.raises(HPACKDecodingError) as error:
        Decoder().decode(data, raw=True)

    assert str(error.value) == f"Variable integer representation is too long: {data[1:]!r}"
    assert error.value.__context__ is None


def test_private_literal_bytes_keep_integer_error() -> None:
    """Byte inputs keep their original diagnostic and exception context."""
    with pytest.raises(HPACKDecodingError) as error:
        Decoder()._decode_literal_index(b"\x40")

    assert str(error.value) == "Unable to decode HPACK integer representation from b''"
    context = error.value.__context__
    assert isinstance(context, IndexError)
    assert context.__context__ is None
