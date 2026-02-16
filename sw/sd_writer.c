/*
 * sd_writer.c — Write files from DDR to SD card via JTAG
 *
 * Usage via JTAG:
 *   1. Power cycle board (SD boot) — FSBL initializes SD controller
 *   2. Halt processor via JTAG
 *   3. dow sd_writer.elf
 *   4. dow -data BOOT_badapple.bin 0x02000000
 *   5. dow -data weights.bin 0x02800000
 *   6. mwr 0x01FFFFF0 <boot_size>    (BOOT.bin size in bytes)
 *   7. mwr 0x01FFFFF4 <weights_size>  (weights.bin size, 0 to skip)
 *   8. con
 */

#include "xil_io.h"
#include "xil_printf.h"
#include "ff.h"

/* DDR layout for file data */
#define BOOT_DATA_ADDR     0x02000000U
#define WEIGHTS_DATA_ADDR  0x02800000U

/* Control block (sizes set by JTAG mwr before run) */
#define BOOT_SIZE_ADDR     0x01FFFFF0U
#define WEIGHTS_SIZE_ADDR  0x01FFFFF4U

static int write_file(const char *name, u32 data_addr, u32 size)
{
    FIL fil;
    FRESULT res;
    UINT bw;
    u32 offset, chunk;

    xil_printf("Writing %s: %u bytes from 0x%08X\r\n", name, size, data_addr);

    res = f_open(&fil, name, FA_WRITE | FA_CREATE_ALWAYS);
    if (res != FR_OK) {
        xil_printf("ERR: open %s failed (%d)\r\n", name, res);
        return -1;
    }

    offset = 0;
    while (offset < size) {
        chunk = size - offset;
        if (chunk > 4096) chunk = 4096;
        res = f_write(&fil, (void *)(data_addr + offset), chunk, &bw);
        if (res != FR_OK || bw != chunk) {
            xil_printf("ERR: write %s at %u failed (res=%d bw=%u)\r\n",
                        name, offset, res, bw);
            f_close(&fil);
            return -1;
        }
        offset += chunk;
        if ((offset & 0xFFFFF) == 0)
            xil_printf("  %uK written\r\n", offset >> 10);
    }

    f_close(&fil);
    xil_printf("%s: %u bytes OK\r\n", name, size);
    return 0;
}

int main(void)
{
    FATFS fs;
    FRESULT res;
    u32 boot_size, weights_size;

    xil_printf("\r\n=== SD Writer ===\r\n");

    boot_size = Xil_In32(BOOT_SIZE_ADDR);
    weights_size = Xil_In32(WEIGHTS_SIZE_ADDR);

    xil_printf("BOOT.bin:    %u bytes at 0x%08X\r\n", boot_size, BOOT_DATA_ADDR);
    xil_printf("weights.bin: %u bytes at 0x%08X\r\n", weights_size, WEIGHTS_DATA_ADDR);

    if (boot_size == 0 || boot_size > 8*1024*1024) {
        xil_printf("ERR: invalid boot size\r\n");
        while (1) ;
    }

    res = f_mount(&fs, "0:", 1);
    if (res != FR_OK) {
        xil_printf("ERR: SD mount failed (%d)\r\n", res);
        while (1) ;
    }
    xil_printf("SD mounted.\r\n");

    if (write_file("BOOT.bin", BOOT_DATA_ADDR, boot_size) < 0) {
        while (1) ;
    }

    if (weights_size > 0 && weights_size <= 8*1024*1024) {
        if (write_file("weights.bin", WEIGHTS_DATA_ADDR, weights_size) < 0) {
            while (1) ;
        }
    }

    f_mount(0, "0:", 0);
    xil_printf("\r\nAll files written. Power cycle to boot.\r\n");
    while (1) ;

    return 0;
}
