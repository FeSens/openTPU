---
name: openroad-image-snapshot
description: >
  Whole-die layout snapshot for any checkpoint: the executable template
  reads an ODB, expands the die box by a margin so boundary pins stay
  visible, and saves a PNG. The Worker runs it after every stage; it can
  also be run standalone on any saved checkpoint.
---

# Whole-die image snapshot

The saved area is the **die box expanded by `img_margin`** (fraction of the
larger die side, default 0.03) so boundary IO pins are not clipped; width is
`img_width` px (default 1024, height keeps the aspect ratio).

Template: `template/snapshot.tcl`, guarded TCL vars:

| var | meaning | default |
|---|---|---|
| `in_odb` | checkpoint to render | required |
| `out_png` | output image path | required |
| `img_width` | save_image -width px | 1024 |
| `img_margin` | fractional pad around the die | 0.03 |

Standalone:

```bash
openroad -no_init -exit rendered_snapshot.tcl   # after rendering the vars
```

Headless note: `save_image` needs `-width` (plain save_image fails with
GUI-0078 under QT_QPA_PLATFORM=minimal).
