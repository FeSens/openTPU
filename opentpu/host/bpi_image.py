"""Bitstream images in the card's BPI x16 flash (docs/flash.md): the byte order the flash holds,
the configuration packets that matter for booting from it, and what a flash dump contains.

Byte order. The FPGA reads the flash 16 bits at a time and takes the first bitstream byte from
D[7:0], the second from D[15:8], each with its bits reversed (UG470: D00 is a byte's MSB). The
.mcs that write_cfgmem -interface BPIx16 writes stores each flash word big-endian, so its bytes
are the bitstream's with the two bytes of every 16-bit word swapped and each byte bit-reversed
("flash order"; checked on deploy_pnbl32_e2521032: the whole otpu.mcs equals that transform of
otpu.bit after its header). The transform is its own inverse. Vivado's readback in bin format
gives flash order; openFPGALoader's --dump-flash undoes the transform (bitstream order). load()
takes either and returns bitstream order.

    python -m opentpu.host.bpi_image info FILE...    .bit, .mcs, .bin or a flash dump: the images
                                                     in it (offset, length, IDCODE, COR0 / COR1,
                                                     WBSTAR, IPROG), sha256
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from dataclasses import dataclass, field

SYNC = bytes.fromhex("aa995566")
_REV = bytes(int(f"{b:08b}"[::-1], 2) for b in range(256))
REGS = {0: "CRC", 1: "FAR", 2: "FDRI", 3: "FDRO", 4: "CMD", 5: "CTL0", 6: "MASK", 7: "STAT",
        8: "LOUT", 9: "COR0", 10: "MFWR", 11: "CBC", 12: "IDCODE", 13: "AXSS", 14: "COR1",
        16: "WBSTAR", 17: "TIMER", 22: "BOOTSTS", 24: "CTL1", 31: "BSPI"}
CMD_IPROG, CMD_DESYNC = 0x0F, 0x0D
_PAD = ((b"\xff" * 4, 2), (bytes.fromhex("11220044"), 1), (bytes.fromhex("000000bb"), 1),
        (b"\xff" * 4, 8))   # before the sync word, backwards: dummy, bus width detection, dummy


def flash_order(data: bytes) -> bytes:
    """bitstream order <-> flash order (odd lengths padded with 0xFF)"""
    if len(data) % 2:
        data = bytes(data) + b"\xff"
    rev = bytes(data).translate(_REV)
    out = bytearray(len(rev))
    out[0::2], out[1::2] = rev[1::2], rev[0::2]
    return bytes(out)


def read_bit(raw: bytes) -> tuple[dict, bytes]:
    """a .bit file -> (header fields a..d: design, part, date, time; the bitstream body)"""
    off = 2 + struct.unpack(">H", raw[:2])[0]
    off += 2                                     # 0x0001
    hdr = {}
    while off < len(raw):
        key = chr(raw[off]); off += 1
        if key == "e":
            n = struct.unpack(">I", raw[off:off + 4])[0]
            return hdr, raw[off + 4:off + 4 + n]
        n = struct.unpack(">H", raw[off:off + 2])[0]
        hdr[key] = raw[off + 2:off + 2 + n].rstrip(b"\0").decode(errors="replace")
        off += 2 + n
    raise ValueError("no bitstream data field in the .bit header")


def read_mcs(text: str) -> bytes:
    """an Intel-hex .mcs -> its bytes from address 0 (gaps 0xFF)"""
    chunks, base = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(":"):
            continue
        rec = bytes.fromhex(line[1:])
        n, addr, typ, data = rec[0], rec[1] << 8 | rec[2], rec[3], rec[4:4 + rec[0]]
        if sum(rec) & 0xFF:
            raise ValueError(f"bad checksum: {line}")
        if typ == 0:
            chunks.append((base + addr, data))
        elif typ == 4:
            base = (data[0] << 8 | data[1]) << 16
        elif typ == 1:
            break
    end = max((a + len(d) for a, d in chunks), default=0)
    img = bytearray(b"\xff" * end)
    for a, d in chunks:
        img[a:a + len(d)] = d
    return bytes(img)


def load(path: str) -> tuple[bytes, str]:
    """a .bit, .mcs or raw .bin / dump -> (bytes in bitstream order, what the file was)"""
    raw = open(path, "rb").read()
    if path.endswith(".bit"):
        hdr, body = read_bit(raw)
        return body, f"bit ({hdr.get('a', '?')}, part {hdr.get('b', '?')}, {hdr.get('c', '')} {hdr.get('d', '')})"
    if path.endswith(".mcs"):
        return flash_order(read_mcs(raw.decode())), "mcs (flash order)"
    fo = raw.find(flash_order(SYNC))
    bo = raw.find(SYNC)
    if fo >= 0 and (bo < 0 or fo < bo):
        return flash_order(raw), "bin (flash order)"
    return raw, "bin (bitstream order)"


@dataclass
class Image:
    offset: int                                  # where the image starts (its padding, bus width words)
    length: int = 0                              # bytes to the end of the DESYNC packet
    writes: list = field(default_factory=list)   # (register, [words]): the settings written

    def reg(self, name: str) -> list:
        return [w for r, w in self.writes if r == name]

    @property
    def iprog(self) -> bool:
        return any(CMD_IPROG in w for w in self.reg("CMD"))


def parse(data: bytes, sync: int) -> Image:
    """the configuration packets from the sync word at `sync`; stops after DESYNC"""
    start = sync       # back over what write_bitstream puts before the sync word, no further
    for word, most in _PAD:                              # (erased flash is 0xFF too)
        for _ in range(most):
            if start < 4 or data[start - 4:start] != word:
                break
            start -= 4
    img = Image(offset=start)
    k, n = sync + 4, len(data) - (len(data) - sync - 4) % 4
    last = None
    while k + 4 <= n:
        w = struct.unpack(">I", data[k:k + 4])[0]; k += 4
        typ = w >> 29
        if typ == 1:
            op, reg, cnt = (w >> 27) & 3, (w >> 13) & 0x1F, w & 0x7FF
            last = reg
            if op == 2 and cnt:
                words = list(struct.unpack(f">{cnt}I", data[k:k + 4 * cnt])); k += 4 * cnt
                if reg not in (0, 1, 2, 10):     # not CRC, FAR, FDRI, MFWR: the settings
                    img.writes.append((REGS.get(reg, str(reg)), words))
                if reg == 4 and CMD_DESYNC in words:
                    img.length = k - start
                    return img
        elif typ == 2:
            k += 4 * (w & 0x7FFFFFF)
            if last not in (2, 3):
                break
        elif w not in (0xFFFFFFFF, 0x20000000):
            break
    img.length = k - start
    return img


def images(data: bytes) -> list[Image]:
    out, i = [], data.find(SYNC)
    while i >= 0:
        img = parse(data, i)
        out.append(img)
        i = data.find(SYNC, max(i + 4, img.offset + img.length))
    return out


def describe(img: Image) -> str:
    def one(name):
        v = img.reg(name)
        return " ".join(f"{x:08x}" for x in v[-1]) if v else "-"
    return (f"offset 0x{img.offset:07x}  length {img.length} ({img.length / 2**20:.2f} MiB)  "
            f"IDCODE {one('IDCODE')}  COR0 {one('COR0')}  COR1 {one('COR1')}  "
            f"WBSTAR {one('WBSTAR')}  IPROG {'yes' if img.iprog else 'no'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m opentpu.host.bpi_image", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["info"])
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    for f in a.files:
        data, kind = load(f)
        used = len(data.rstrip(b"\xff"))
        print(f"{f}: {kind}, {len(data)} bytes ({used} before the trailing 0xFF), "
              f"sha256 {hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]}")
        found = images(data)
        for img in found:
            print("  " + describe(img))
        if not found:
            print("  no bitstream (no sync word)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
