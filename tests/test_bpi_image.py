"""opentpu/host/bpi_image.py: the BPI x16 flash byte order, .bit / .mcs / dump parsing, and the
configuration settings found in an image (COR0, WBSTAR, IPROG)."""
import struct

from opentpu.host import bpi_image as B


def bitstream(wbstar=None, cor0=0x02003FE5, frames=3):
    w = [0xFFFFFFFF] * 8 + [0x000000BB, 0x11220044, 0xFFFFFFFF, 0xFFFFFFFF, 0xAA995566,
                            0x20000000, 0x30012001, cor0, 0x30018001, 0x03751093]
    if wbstar is not None:
        w += [0x30020001, wbstar, 0x30008001, B.CMD_IPROG, 0x20000000]
    w += [0x30004000, 0x50000000 | frames] + [0x12345678] * frames
    w += [0x30008001, B.CMD_DESYNC] + [0x20000000] * 4
    return struct.pack(f">{len(w)}I", *w)


def bit_file(body):
    def field(key, s):
        s = s.encode() + b"\0"
        return key.encode() + struct.pack(">H", len(s)) + s
    return (struct.pack(">H", 9) + bytes.fromhex("0ff00ff00ff00ff000") + struct.pack(">H", 1)
            + field("a", "otpu_fpga_top;COMPRESS=TRUE") + field("b", "7k480tffg1156")
            + field("c", "2026/09/28") + field("d", "10:32:50")
            + b"e" + struct.pack(">I", len(body)) + body)


def mcs_text(data, base=0):
    lines = []
    for off in range(0, len(data), 16):
        a = base + off
        if off == 0 or a % 0x10000 == 0:
            rec = bytes([2, 0, 0, 4, a >> 24 & 0xFF, a >> 16 & 0xFF])
            lines.append(":" + (rec + bytes([-sum(rec) & 0xFF])).hex().upper())
        chunk = data[off:off + 16]
        rec = bytes([len(chunk), a >> 8 & 0xFF, a & 0xFF, 0]) + chunk
        lines.append(":" + (rec + bytes([-sum(rec) & 0xFF])).hex().upper())
    return "\n".join(lines + [":00000001FF"]) + "\n"


def test_flash_order_is_what_write_cfgmem_writes():
    # the bus width words as deploy_pnbl32_e2521032's otpu.bit and otpu.mcs hold them
    assert B.flash_order(bytes.fromhex("000000bb11220044")) == bytes.fromhex("0000dd0044882200")
    assert B.flash_order(B.SYNC) == bytes.fromhex("995566aa")
    body = bitstream()
    assert B.flash_order(B.flash_order(body)) == body


def test_bit_header_and_settings(tmp_path):
    body = bitstream(wbstar=0x00900000)
    f = tmp_path / "g.bit"
    f.write_bytes(bit_file(body))
    data, kind = B.load(str(f))
    assert data == body and "7k480tffg1156" in kind
    (img,) = B.images(data)
    assert img.offset == 0 and img.length == len(body) - 16      # to the end of DESYNC
    assert img.iprog and img.reg("WBSTAR") == [[0x00900000]]
    assert img.reg("IDCODE") == [[0x03751093]] and img.reg("COR0") == [[0x02003FE5]]
    assert "IPROG yes" in B.describe(img)
    assert not B.images(bitstream())[0].iprog


def test_mcs_and_flash_dump(tmp_path):
    a, b = bitstream(wbstar=0x4000), bitstream(cor0=0x02007FE5, frames=50)
    (tmp_path / "a.mcs").write_text(mcs_text(B.flash_order(a)))
    assert B.load(str(tmp_path / "a.mcs"))[0] == a
    dump = bytearray(b"\xff" * 0x20000)                         # a in slot 0, b at 0x8000 bytes
    dump[:len(a)] = B.flash_order(a)
    dump[0x8000:0x8000 + len(b)] = B.flash_order(b)
    (tmp_path / "dump.bin").write_bytes(dump)
    data, kind = B.load(str(tmp_path / "dump.bin"))
    assert kind == "bin (flash order)"
    imgs = B.images(data)
    assert [i.offset for i in imgs] == [0, 0x8000]
    assert imgs[1].reg("COR0") == [[0x02007FE5]]
    (tmp_path / "plain.bin").write_bytes(B.flash_order(dump))  # openFPGALoader's dump order
    assert B.load(str(tmp_path / "plain.bin"))[1] == "bin (bitstream order)"


def test_cli(tmp_path, capsys):
    f = tmp_path / "g.bit"
    f.write_bytes(bit_file(bitstream(wbstar=0x00900000)))
    assert B.main(["info", str(f)]) == 0
    out = capsys.readouterr().out
    assert "WBSTAR 00900000" in out and "IPROG yes" in out
