"""opentpu/host/cardimage.py, the image a BCU1525 holds: the accelerator's own registers in RUN;
in SAFE (they are fenced) the governor's identity words on a build with them, else the record a
bring-up left, which counts only under the boot it was written in; otpu-smi's Image rows."""
import json

import pytest

from opentpu.host import cardimage as CI
from opentpu.host import gov as G
from opentpu.host import smi
from opentpu.host.fake import FakeBcu
from opentpu.target import arch_hash, build_config, load

DEV = "/dev/fakebcu0"


@pytest.fixture
def state(tmp_path, monkeypatch):
    """A state directory and a boot id of the test's own."""
    boot = tmp_path / "boot_id"
    boot.write_text("boot-1\n")
    monkeypatch.setattr(CI, "BOOT_ID", str(boot))
    monkeypatch.setenv("BCU_BUILD", str(tmp_path / "build"))
    return boot


def test_a_running_card_answers_for_itself(state):
    fk = FakeBcu(run=True)
    img = CI.held(fk, DEV)
    h = arch_hash(build_config(load("bcu1525-wg")))
    assert img == {"target": "bcu1525-wg", "platform": load("bcu1525-wg").platform_id,
                   "slices": 1, "cards": 1, "cfg_hash": h, "build_id": 0x1234ABCD,
                   "core_khz": 150000, "source": "registers"}
    assert CI.kind(img) == "one" and fk.writes == []
    assert CI.text(img) == (f"bcu1525-wg  S=1  (the card's registers)  build 1234abcd  "
                            f"CFG_HASH {h:08x}  150 MHz")


def test_in_safe_a_build_without_the_words_is_unknown_until_a_bring_up(state):
    """SAFE on today's builds: nothing says (the registers fenced, IMAGE reads 0); the record a
    bring-up wrote while the card ran answers afterwards, with when it was read."""
    safe = FakeBcu()
    assert G.Gov(safe).image() is None
    assert CI.held(safe, DEV) == {"source": None} and CI.kind(CI.held(safe, DEV)) == "unknown"
    assert CI.note(safe, DEV) is None                       # (fenced: nothing to record)
    rec = CI.note(FakeBcu(run=True), DEV)                   # the bring-up's read
    assert rec["boot_id"] == "boot-1" and rec["target"] == "bcu1525-wg"
    f = CI.state_dir() / "image-fakebcu0.json"
    assert json.loads(f.read_text()) == rec
    img = CI.held(safe, DEV)
    assert img["source"] == "record" and img["when"] == rec["when"]
    assert {k: img[k] for k in CI.FIELDS} == {k: rec[k] for k in CI.FIELDS}
    assert "as read at the last bring-up" in CI.text(img) and CI.kind(img) == "one"
    assert safe.writes == []


def test_a_record_counts_only_under_its_boot_and_with_a_governor(state):
    CI.note(FakeBcu(run=True), DEV)
    safe = FakeBcu()
    assert CI.held(safe, DEV)["source"] == "record"
    assert CI.held(safe, "/dev/fakebcu1") == {"source": None}       # (another card's file)
    state.write_text("boot-2\n")                           # a reboot: the image may be another
    assert CI.record(DEV) is None and CI.held(safe, DEV) == {"source": None}
    state.write_text("boot-1\n")
    assert CI.held(safe, DEV)["source"] == "record"

    class NoGov:                                            # (no BCU image: its ID is not GOV1)
        def reg_read(self, a):
            return 0xFFFFFFFF
    assert CI.held(NoGov(), DEV) == {"source": None}
    state.unlink()                                          # a host that cannot tell its boot
    assert CI.held(safe, DEV) == {"source": None}
    state.write_text("boot-1\n")
    (CI.state_dir() / "image-fakebcu0.json").write_text("{")
    assert CI.held(safe, DEV) == {"source": None}           # a damaged record
    CI.note(FakeBcu(run=True), DEV)
    CI.forget(DEV)
    assert CI.held(safe, DEV) == {"source": None}
    CI.forget(DEV)                                          # (none: no error)


def test_the_governors_words_answer_in_safe(state):
    """A build whose governor holds the identity: SAFE answers without a record, and the words
    are the accelerator's own."""
    safe, run = FakeBcu(gov_image=True), FakeBcu(run=True, gov_image=True)
    g = G.Gov(safe).image()
    assert g["cards"] == 1 and g["slices"] == 1 and g["core_khz"] == 150000
    img = CI.held(safe, DEV)
    assert img["source"] == "governor"
    live = CI.held(run, DEV)
    assert live["source"] == "registers"
    assert {k: img[k] for k in CI.FIELDS} == {k: live[k] for k in CI.FIELDS}
    assert CI.text(img).endswith("150 MHz") and "(the governor)" in CI.text(img)


def test_a_card_of_the_two_card_machine_and_a_variant_by_its_hash():
    """The CFG_HASH names the variant (a variant shares its target's platform id); a card of
    bcu1525-x2 is one of two; a hash no build of this tree has keeps the target with a `?`."""
    t = load("bcu1525-x2:st16")
    h = arch_hash(build_config(t))
    assert CI.target_name(t.platform_id, 2, h) == ("bcu1525-x2:st16", 2)
    img = CI.from_gov({"platform": t.platform_id, "slices": 2, "cards": 2, "cfg_hash": h,
                       "build_id": 0xABCD0123, "core_khz": 150000})
    assert CI.kind({**img, "source": "governor"}) == "x2"
    assert CI.texts({**img, "source": "governor"}) == (
        "bcu1525-x2:st16  S=2 of 2 cards  (the governor)",
        f"build abcd0123  CFG_HASH {h:08x}  150 MHz")
    assert CI.target_name(t.platform_id, 2, 0x12345678) == ("bcu1525-x2:?", 2)
    assert CI.target_name(0xFFF, 1, 0) == (None, 1)
    assert CI.from_gov({"platform": 0, "slices": 0, "cards": 0, "cfg_hash": 0, "build_id": 0,
                        "core_khz": 100000}) is None       # (a top without an accelerator)
    assert CI.from_gov(None) is None


def test_otpu_smi_shows_the_image_in_safe_and_in_run(state, tmp_path, capsys):
    none = str(tmp_path / "none.json")
    args = ["--dev", DEV, "--power-json", none, "-i", "0"]

    def tab(fk):
        assert smi.main(args, open_transport=lambda dev: fk) == 0
        out = capsys.readouterr().out
        assert all(len(r) == smi.W for r in out.splitlines()[1:]), out
        return out
    assert "Image      unknown until a bring-up" in tab(FakeBcu())
    out = tab(FakeBcu(run=True))
    assert "Image      bcu1525-wg  S=1  (the card's registers)" in out
    assert "build 1234abcd  CFG_HASH " in out and "150 MHz" in out
    CI.note(FakeBcu(run=True), DEV)
    out = tab(FakeBcu())
    assert "Image      bcu1525-wg  S=1  (as read at the last bring-up " in out
    out = tab(FakeBcu(gov_image=True))
    assert "Image      bcu1525-wg  S=1  (the governor)" in out and "build 1234abcd" in out


def test_the_module_prints_the_wrappers_word(state, capsys, monkeypatch):
    from opentpu.host import board as BD
    fk = FakeBcu(gov_image=True)
    monkeypatch.setattr(BD, "XdmaTransport", lambda dev, dma=True: fk)
    assert CI.main(["--dev", DEV]) == 0
    assert capsys.readouterr().out.startswith("one  bcu1525-wg  S=1  (the governor)")
    assert CI.main(["--dev", DEV, "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["kind"] == "one" and d["source"] == "governor"
    fk2 = FakeBcu()
    monkeypatch.setattr(BD, "XdmaTransport", lambda dev, dma=True: fk2)
    CI.main(["--dev", DEV])
    assert capsys.readouterr().out.strip() == "unknown  unknown until a bring-up"


def test_a_card_of_the_two_card_machine_names_its_variant():
    """A card of bcu1525-x2 reports its own two slices and that build's CFG_HASH: the variant is
    found by the hash of the card's build, not the four-slice machine's."""
    from opentpu import target as T
    from opentpu.host import cardimage as CI
    pid = T.load("bcu1525-x2").platform_id
    assert CI.target_name(pid, 2, 0xe331f79c) == ("bcu1525-x2:st16", 2)
    assert CI.target_name(pid, 2, 0x12345678)[0] == "bcu1525-x2:?"
