/* One memory at 0 (code, constants, .bss and the stack): the image is its initial content and
   the CPU restarts in place, so there is no initialised writable data. */
OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { mem : ORIGIN = 0, LENGTH = MEM_BYTES }
SECTIONS
{
    .text : { *(.text.start) *(.text .text.*) } > mem
    .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > mem
    .data : { *(.data .data.* .sdata .sdata.*) } > mem
    ASSERT(SIZEOF(.data) == 0, "initialised writable data: the CPU restarts without reloading it")
    .bss (NOLOAD) : ALIGN(4) {
        _bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > mem
    _stack_top = ORIGIN(mem) + LENGTH(mem);
    ASSERT(_stack_top - _bss_end >= 1024, "less than 1 KiB left for the stack")
    /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.note*) *(.eh_frame*) }
}
