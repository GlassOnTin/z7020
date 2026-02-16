/*
 * sd_writer.c — Tiny helper to write BOOT.bin from DDR to SD card
 *
 * Usage via JTAG:
 *   1. Download this ELF
 *   2. Download BOOT.bin data to DDR at DATA_ADDR using: dow -data <file> <addr>
 *   3. Set r0 = file size in bytes
 *   4. Run
 */

#include "xil_io.h"
#include "xil_printf.h"
#include "ff.h"

#define DATA_ADDR  0x02000000U  /* Where BOOT.bin data is loaded in DDR */
#define SIZE_ADDR  0x01FFFFF0U  /* Where file size is stored (set via mwr) */

int main(void)
{
    FATFS fs;
    FIL fil;
    FRESULT res;
    UINT bw;
    u32 size;
    u32 offset;
    u32 chunk;

    xil_printf("\r\n=== SD Writer ===\r\n");

    size = Xil_In32(SIZE_ADDR);
    xil_printf("File size: %u bytes at 0x%08X\r\n", size, DATA_ADDR);

    if (size == 0 || size > 8*1024*1024) {
        xil_printf("ERR: invalid size (write size to 0x%08X first)\r\n", SIZE_ADDR);
        while (1) ;
    }

    res = f_mount(&fs, "0:", 1);
    if (res != FR_OK) {
        xil_printf("ERR: SD mount failed (%d)\r\n", res);
        while (1) ;
    }
    xil_printf("SD mounted.\r\n");

    res = f_open(&fil, "BOOT.bin", FA_WRITE | FA_CREATE_ALWAYS);
    if (res != FR_OK) {
        xil_printf("ERR: open BOOT.bin for write failed (%d)\r\n", res);
        while (1) ;
    }

    /* Write in 4KB chunks */
    offset = 0;
    while (offset < size) {
        chunk = size - offset;
        if (chunk > 4096) chunk = 4096;
        res = f_write(&fil, (void *)(DATA_ADDR + offset), chunk, &bw);
        if (res != FR_OK || bw != chunk) {
            xil_printf("ERR: write at offset %u failed (res=%d bw=%u)\r\n",
                        offset, res, bw);
            f_close(&fil);
            while (1) ;
        }
        offset += chunk;
        if ((offset & 0xFFFFF) == 0)
            xil_printf("  %uK written\r\n", offset >> 10);
    }

    f_close(&fil);
    f_mount(0, "0:", 0);

    xil_printf("BOOT.bin written: %u bytes\r\n", size);
    xil_printf("Done. Power cycle to boot.\r\n");
    while (1) ;

    return 0;
}
